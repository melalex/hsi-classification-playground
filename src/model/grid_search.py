import csv

import numpy as np
import torch
import tqdm
import multiprocessing as mp

from abc import ABC
from itertools import product
from pathlib import Path
from typing import Optional, Sequence
from src.util.list_ext import split


class GridSearchAdapter[M](ABC):

    def params_grid(self) -> dict[str, Sequence[float]]:
        pass

    def init_model(self, params: dict[str, float]):
        pass

    def fit_model(self, model: M):
        pass

    def score_model(self, model: M) -> list[dict[str, float]]:
        pass


class GridSearch[M]:
    adapter: GridSearchAdapter[M]
    optimize_metric: str
    log_dir: Optional[Path]
    num_workers: int

    def __init__(
        self,
        adapter: GridSearchAdapter[M],
        optimize_metric: str,
        log_dir: Optional[Path],
        num_workers: int = 1,
    ):
        self.adapter = adapter
        self.optimize_metric = optimize_metric
        self.log_dir = log_dir
        self.num_workers = num_workers

    def run(
        self, start_from: dict[int, int] = {}
    ) -> tuple[M, dict[str, Sequence[float]], dict[str, float]]:
        def resume_split_from(split: int):
            if split in start_from:
                return start_from[split]
            else:
                return 0

        if torch.cuda.is_available():
            mp.set_start_method("spawn", force=True)

        param_grid = self.adapter.params_grid()
        keys = list(param_grid.keys())
        params_list = list(product(*param_grid.values()))

        with mp.Pool(processes=self.num_workers) as pool:
            param_splits = split(params_list, self.num_workers)

            params = [
                (
                    self,
                    i,
                    param_splits[i],
                    len(param_splits[i]),
                    resume_split_from(i),
                    keys,
                )
                for i in range(self.num_workers)
            ]

            result = pool.starmap(run_split_multiprocessing_workaround, params)

            return max(result, key=lambda x: x[2][self.optimize_metric])

    def run_split(
        self,
        split: int,
        params_list: list[dict[str, float]],
        split_len: int,
        start_from: int,
        keys: Sequence[str],
    ) -> tuple[M, dict[str, Sequence[float]], dict[str, float]]:
        params_list = params_list[start_from:]
        best_score = {self.optimize_metric: -np.inf}
        best_params = None
        best_model = None

        with tqdm.tqdm(initial=start_from, total=split_len) as pb:
            for values in params_list:
                params = dict(zip(keys, values))
                model = self.adapter.init_model(split, params)
                self.adapter.fit_model(model)
                score = self.adapter.score_model(model)

                self.__log_result(split, params, score)

                if score[-1][self.optimize_metric] > best_score[self.optimize_metric]:
                    best_score = score[-1]
                    best_params = params
                    best_model = model

                pb.set_postfix(split=split, best_score=best_score[self.optimize_metric])
                pb.update()

        return best_model, best_params, best_score

    def __log_result(self, split, params, score):
        if self.log_dir is None:
            return

        log_file = self.log_dir / f"split_{split}.csv"

        log_file.parent.mkdir(exist_ok=True, parents=True)
        file_exists = log_file.exists()
        last_score = score[-1]
        best_score_idx = max(
            enumerate(score), key=lambda x: x[1][self.optimize_metric]
        )[0]
        best_score = score[best_score_idx]

        csv_row = {
            **params,
            **last_score,
            "best_iteration": best_score_idx,
            f"best_{self.optimize_metric}": best_score[self.optimize_metric],
        }

        with open(log_file, mode="a") as f:
            writer = csv.DictWriter(f, fieldnames=csv_row.keys())

            if not file_exists:
                writer.writeheader()

            writer.writerows([csv_row])


def run_split_multiprocessing_workaround[M](
    instance: GridSearch[M],
    split: int,
    params_list: list[dict[str, float]],
    split_len: int,
    start_from: int,
    keys: Sequence[str],
):
    return instance.run_split(split, params_list, split_len, start_from, keys)

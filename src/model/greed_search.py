import csv

import torch
import tqdm
import multiprocessing as mp

from abc import ABC
from pathlib import Path
from typing import Optional, Sequence


class GreedSearchAdapter[M](ABC):

    def init_interval(self) -> list[tuple[int, int]]:
        pass

    def get_params(self, i: int) -> dict[str, float]:
        pass

    def init_model(self, split, params: dict[str, float]):
        pass

    def fit_model(self, model: M):
        pass

    def score_model(self, model: M) -> list[dict[str, float]]:
        pass


class GreedSearch[M]:
    adapter: GreedSearchAdapter[M]
    optimize_metric: str
    log_dir: Optional[Path]
    num_workers: int

    def __init__(
        self,
        adapter: GreedSearchAdapter[M],
        optimize_metric: str,
        log_dir: Optional[Path],
    ):
        self.adapter = adapter
        self.optimize_metric = optimize_metric
        self.log_dir = log_dir

    def run(
        self, start_from: dict[int, tuple[int, int]] = {}
    ) -> tuple[M, dict[str, Sequence[float]], dict[str, float]]:
        split_intervals = self.adapter.init_interval()
        num_workers = len(split_intervals)

        def resume_split_from(split: int) -> tuple[int, int]:
            if split in start_from:
                return start_from[split]
            else:
                return split_intervals[split]

        if torch.cuda.is_available():
            mp.set_start_method("spawn", force=True)

        if num_workers == 1:
            return run_split_multiprocessing_workaround(self, 0, resume_split_from(0))

        with mp.Pool(processes=num_workers) as pool:
            params = [
                (
                    self,
                    i,
                    resume_split_from(i),
                )
                for i in range(num_workers)
            ]

            result = pool.starmap(run_split_multiprocessing_workaround, params)

            return max(result, key=lambda x: x[2][self.optimize_metric])

    def run_split(
        self, split: int, low: int, high: int
    ) -> tuple[M, dict[str, float], dict[str, float]]:
        with tqdm.tqdm() as pb:
            optimize = self.optimize_metric

            low_model = None
            high_model = None
            postfix = {}

            while low < high:
                mid = (low + high) // 2
                next_to_mid = mid + 1

                mid_params = self.adapter.get_params(mid)
                mid_model, mid_score = self.__score_model(split, mid_params)

                next_to_mid_params = self.adapter.get_params(next_to_mid)
                next_to_mid_model, next_to_mid_score = self.__score_model(
                    split, next_to_mid_params
                )

                if mid_score[-1][optimize] < next_to_mid_score[-1][optimize]:
                    low = next_to_mid
                    low_params = next_to_mid_params
                    low_model = next_to_mid_model
                    low_score = next_to_mid_score

                    postfix["low"] = low
                    postfix["low_score"] = low_score[-1][optimize]

                    self.__log_result(split, low_params, low_score)
                else:
                    high = mid
                    high_params = mid_params
                    high_model = mid_model
                    high_score = mid_score

                    postfix["high"] = high
                    postfix["high_score"] = high_score[-1][optimize]

                    self.__log_result(split, high_params, high_score)

                pb.set_postfix(**postfix)
                pb.update()

            if low_model:
                return low_model, low_params, low_score
            else:
                return high_model, high_params, high_score

    def __score_model(
        self, split: int, params: dict[str, float]
    ) -> tuple[M, list[dict[str, float]]]:
        model = self.adapter.init_model(split, params)
        self.adapter.fit_model(model)
        score = self.adapter.score_model(model)

        return model, score

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
    instance: GreedSearch[M], split: int, interval: tuple[int, int]
):
    low, high = interval
    return instance.run_split(split, low, high)

from abc import ABC
import csv
from itertools import product
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from src.util.progress_bar import create_progress_bar


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
    log_file: Optional[Path]

    def __init__(
        self,
        adapter: GridSearchAdapter[M],
        optimize_metric: str,
        log_file: Optional[Path],
    ):
        self.adapter = adapter
        self.optimize_metric = optimize_metric
        self.log_file = log_file

    def run(self, start_from=0) -> tuple[M, dict[str, Sequence[float]], float]:
        best_score = {self.optimize_metric: -np.inf}
        best_params = None
        best_model = None
        param_grid = self.adapter.params_grid()
        keys = list(param_grid.keys())
        params_list = list(product(*param_grid.values()))
        init_params_list_len = len(params_list)
        params_list = params_list[start_from:]

        with create_progress_bar()(
            initial=start_from, total=init_params_list_len
        ) as pb:
            for values in params_list:
                params = dict(zip(keys, values))
                model = self.adapter.init_model(params)
                self.adapter.fit_model(model)
                score = self.adapter.score_model(model)

                self.__log_result(params, score)

                if score[-1][self.optimize_metric] > best_score[self.optimize_metric]:
                    best_score = score[-1]
                    best_params = params
                    best_model = model

                pb.set_postfix(best_score=best_score[self.optimize_metric])
                pb.update()

        return best_model, best_params, best_score

    def __log_result(self, params: dict[str, float], score: list[dict[str, float]]):
        if self.log_file is None:
            return

        self.log_file.parent.mkdir(exist_ok=True, parents=True)
        file_exists = self.log_file.exists()
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

        with open(self.log_file, mode="a") as f:
            writer = csv.DictWriter(f, fieldnames=csv_row.keys())

            if not file_exists:
                writer.writeheader()

            writer.writerows([csv_row])

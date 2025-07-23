from abc import ABC
import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import torch.utils.data as data
import tqdm


@dataclass
class HistoryEntry:
    id: str
    score: dict[str, float]


class RepresentativeDatapointsSearchAdapter[M](ABC):

    def get_run_params(
        self,
    ) -> Sequence[
        tuple[str, data.DataLoader, Optional[data.DataLoader], dict[str, float]]
    ]:
        pass

    def init_model(self, params: dict[str, float]) -> M:
        pass

    def fit_model(
        self, model: M, loader: data.DataLoader, eval_loader: Optional[data.DataLoader]
    ) -> list[dict[str, float]]:
        pass

    def score_model(self, model: M) -> dict[str, float]:
        pass

    def on_scored_model(
        self,
        id: str,
        params: dict[str, float],
        model: M,
        model_history: list[dict[str, float]],
        score: dict[str, float],
    ):
        pass


class RepresentativeDatapointsSearch[M]:
    adapter: RepresentativeDatapointsSearchAdapter[M]
    log_dir: Optional[Path]
    run_name: Optional[str]

    def __init__(
        self,
        adapter: RepresentativeDatapointsSearchAdapter[M],
        log_dir: Optional[Path] = None,
        run_name: Optional[Path] = None,
    ):
        self.adapter = adapter
        self.log_dir = log_dir
        self.run_name = run_name

    def run(self) -> list[HistoryEntry]:
        params_list = self.adapter.get_run_params()
        history = []

        with tqdm.tqdm(total=len(params_list)) as pb:
            for id, dl, eval_dl, params in params_list:
                model = self.adapter.init_model(params)
                model_history = self.adapter.fit_model(model, dl, eval_dl)
                score = self.adapter.score_model(model)

                self.adapter.on_scored_model(id, params, model, model_history, score)

                self.__log_result(id, params, score)
                self.__log_history(id, model_history)

                history.append(HistoryEntry(id, score))

                pb.set_postfix(id=id, **score)
                pb.update()

        return history

    def __log_result(self, id, params, score):
        if self.log_dir is None:
            return

        file_name = (
            f"{self.run_name}.csv" if self.run_name else "representative-datapoints.csv"
        )
        log_file = self.log_dir / file_name

        log_file.parent.mkdir(exist_ok=True, parents=True)
        file_exists = log_file.exists()
        
        csv_row = {
            "id": id,
            "params": params,
            **score,
        }

        with open(log_file, mode="w") as f:
            writer = csv.DictWriter(f, fieldnames=csv_row.keys())

            if not file_exists:
                writer.writeheader()

            writer.writerows([csv_row])

        return log_file

    def __log_history(self, id, model_history):
        if self.log_dir is None:
            return

        history_file_name = (
            f"{self.run_name}-{id}_history.csv"
            if self.run_name
            else f"representative-datapoints-{id}_history.csv"
        )

        history_file = self.log_dir / history_file_name

        self.log_dir.mkdir(exist_ok=True, parents=True)

        with open(history_file, mode="w") as f:
            writer = csv.DictWriter(f, fieldnames=model_history[0].keys())

            writer.writeheader()
            writer.writerows(model_history)

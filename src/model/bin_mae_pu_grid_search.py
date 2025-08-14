import torch
import torch.utils.data as data

from dataclasses import dataclass
from typing import Any, Optional, Sequence

from src.model.autoencoder import AutoEncoderWithClassificationHead
from src.model.dbda import DBDA
from src.model.finetune import SimpleBinClassificationHead
from src.model.grid_search import GridSearchAdapter
from src.model.vit_hsi_decoder import ViTHsiDecoder
from src.trainer.base_trainer import (
    AdamOptimizedModule,
    BaseTrainer,
    TrainableModule,
    TrainerFeedback,
)
from src.trainer.masked_autoencoder_trainer import HsiPatchMasking
from src.trainer.masked_autoencoder_with_classification_trainer import (
    MaskedAutoEncoderWithClasificationTrainer,
)
from src.util.loss import (
    MaskedAutoencoderLoss,
    MaskedAutoencoderWithClassificationHeadLoss,
    PuLoss,
)


@dataclass
class BinMaePuPipeline:
    trainer: BaseTrainer
    model: TrainableModule
    feedback: Optional[TrainerFeedback] = None


class BinMaePuSearchAdapter(GridSearchAdapter[BinMaePuPipeline]):

    def __init__(
        self,
        params: dict[str, Sequence[Any]],
        train_dataloader: data.DataLoader,
        eval_dataloader: data.DataLoader,
        positive_prob: float,
        device: torch.device,
    ):
        self.params = params
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.positive_prob = positive_prob
        self.device = device

    def params_grid(self) -> dict[str, Sequence[Any]]:
        return self.params

    def init_model(self, split, params: dict[str, float]):
        torch.cuda.empty_cache()

        def extract_prediction(y_hat):
            return (torch.sigmoid(y_hat) > params["prediction_threshold"]).to(
                dtype=torch.int
            )

        pu_loss = PuLoss(
            prior=self.positive_prob,
            nnpu=True,
            loss_fn=params["loss_fun"],
        )

        encoder = DBDA(
            band=params["target_dim"],
            classes=params["latent_space_size"],
        )

        decoder = ViTHsiDecoder(
            latent_dim=params["latent_space_size"],
            out_h=params["patch_size"],
            out_w=params["patch_size"],
            out_bands=params["target_dim"],
            embed_dim=params["decoder_embed_dim"],
            n_layers=params["decoder_layers"],
            n_heads=params["decoder_heads"],
            mlp_dim=params["decoder_mlp_dim"],
        )

        cls_head = SimpleBinClassificationHead(params["latent_space_size"])

        model = AdamOptimizedModule(
            net=AutoEncoderWithClassificationHead(encoder, decoder, cls_head),
            lr=params["learning_rate"],
            weight_decay=params["weight_decay"],
        )

        trainer = MaskedAutoEncoderWithClasificationTrainer(
            loss_fun=MaskedAutoencoderWithClassificationHeadLoss(
                autoencoder_loss=MaskedAutoencoderLoss(
                    alpha=params["loss_alpha"], beta=params["loss_beta"]
                ),
                cls_loss=pu_loss,
                autoencoder_loss_weight=params["autoencoder_loss_weight"],
                cls_loss_weight=params["cls_loss_weight"],
            ),
            num_classes=2,
            epochs=params["num_epochs"],
            masking=HsiPatchMasking(
                mask_ratio=params["mask_ratio"],
                mode=params["mask_mode"],
                fill_value=["fil_value"],
            ),
            device=self.device,
            extract_prediction=extract_prediction,
        )

        return BinMaePuPipeline(trainer=trainer, model=model)

    def fit_model(self, model: BinMaePuPipeline):
        feedback, _ = model.trainer.fit(
            model=model.model,
            train_dataloader=self.train_dataloader,
            eval_dataloader=self.eval_dataloader,
        )
        model.feedback = feedback

    def score_model(self, model: BinMaePuPipeline) -> list[dict[str, float]]:
        return [it.eval for it in model.feedback.history]

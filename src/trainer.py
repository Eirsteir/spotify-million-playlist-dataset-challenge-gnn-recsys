import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torchmetrics import MeanSquaredError, RetrievalRPrecision, RetrievalNormalizedDCG

from src.models import IGMC


def num_graphs(data):
    if data.batch is not None:
        return data.num_graphs
    else:
        return data.x.size(0)


class LightningIGMC(LightningModule):
    def __init__(
            self,
            lr,
            lr_decay_factor,
            lr_decay_step_size,
            ARR,
            num_features,
            num_relations=1, 
            latent_dim=[32, 32, 32, 32], 
            num_layers=4, 
            num_bases=2, 
            adj_dropout=0.2,
            force_undirected=False, 
            use_features=False, 
            n_side_features=0
        ):
        super().__init__()
        self.save_hyperparameters()

        self.lr = lr
        self.lr_decay_factor = lr_decay_factor
        self.lr_decay_step_size = lr_decay_step_size
        self.ARR = ARR

        self.model = IGMC(
            num_features,
            num_relations, 
            latent_dim, 
            num_layers, 
            num_bases, 
            adj_dropout,
            force_undirected, 
            use_features, 
            n_side_features
        )

        self.train_rmse = MeanSquaredError(squared=False)
        
        self.val_rmse = MeanSquaredError(squared=False)
        self.val_ndcg = RetrievalNormalizedDCG()

        self.test_rmse = MeanSquaredError(squared=False)
        self.test_ndcg = RetrievalNormalizedDCG()

        self.criterion = F.mse_loss

    def forward(self, data):
        return self.model(data)

    def training_step(self, batch, batch_idx: int):
        out = self(batch)
        loss = self.criterion(out, batch.y.view(-1))

        if self.ARR != 0:
            for gconv in self.model.convs:
                w = torch.matmul(
                    gconv.comp,
                    gconv.weight.view(gconv.num_bases, -1)
                ).view(gconv.num_relations, gconv.in_channels, gconv.out_channels)
                reg_loss = torch.sum((w[1:, :, :] - w[:-1, :, :]) ** 2)
                loss += self.ARR * reg_loss

        self.train_rmse(out, batch.y.view(-1))
        
        self.log('train_rmse', self.train_rmse, on_step=False, on_epoch=True,
                 prog_bar=True, sync_dist=True)

        return loss * num_graphs(batch)  # TODO: remove?

    def validation_step(self, batch, batch_idx: int):
        out = self(batch)
        loss = self.criterion(out, batch.y.view(-1), reduction="sum")

        self.val_rmse(out, batch.y.view(-1))
        self.val_ndcg(out, batch.y.view(-1), indexes=batch.y.view(-1).new_zeros(batch.y.view(-1).shape).long())

        self.log('val_rmse', self.val_rmse, on_step=False, on_epoch=True,
                 prog_bar=True, sync_dist=True)
        self.log('val_ndcg', self.val_ndcg, on_step=False, on_epoch=True,
                 prog_bar=True, sync_dist=True)
                 
    def test_step(self, batch, batch_idx: int):
        out = self(batch)
        loss = self.criterion(out, batch.y.view(-1), reduction="sum")

        self.test_rmse(out, batch.y.view(-1))

        self.log('test_rmse', self.test_rmse, on_step=False, on_epoch=True,
                 prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.lr_decay_step_size,
                                                    gamma=self.lr_decay_factor)
        return [optimizer], [scheduler]

    # def parameters(self):
    #     return self.model.parameters()
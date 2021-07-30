import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F


class Print(nn.Module):
    """Debugging only"""

    def forward(self, x):
        print(x.size())
        return x


class Clamp(nn.Module):
    """Clamp energy output"""

    def forward(self, x):
        x = torch.clamp(x, min=0, max=30)
        return x


class SimpleConv(pl.LightningModule):
    def __init__(self, mode: ["classification", "regression"] = "classification"):
        super().__init__()
        self.mode = mode
        self.layer1 = nn.Sequential(
                    nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2),
                    nn.BatchNorm2d(6),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2),
                    nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=2),
                    nn.BatchNorm2d(16),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2),
                    nn.Flatten(),
                )

        self.drop_out = nn.Dropout()
        self.selu = nn.SELU()
        
        self.fc1 = nn.Linear(14400, 1800)
        self.fc7 = nn.Linear(1800, 500) #nn.Linear(900, 500)
        self.fc2 = nn.Linear(500, 2)  # for classification
        self.fc3 = nn.Linear(500, 1)  # for regression


        self.stem = nn.Sequential(
            self.layer1, self.drop_out,
            self.fc1,self.selu,self.drop_out,
            self.fc7,self.selu,self.drop_out,
            )
        if self.mode == "classification":
            self.classification = nn.Sequential(self.stem, self.fc2)
        else:
            self.regression = nn.Sequential(self.stem, self.fc3)

        self.train_acc = pl.metrics.Accuracy()
        self.valid_acc = pl.metrics.Accuracy()
        self.test_acc = pl.metrics.Accuracy()

    def training_step(self, batch, batch_idx):
        # --------------------------
        x_target, class_target, reg_target, _ = batch
        if self.mode == "classification":
            class_pred = self.classification(x_target.float())
            class_loss = F.binary_cross_entropy_with_logits(
                class_pred, class_target.float()
            )
            self.train_acc(torch.sigmoid(class_pred), class_target)
            self.log("train_acc", self.train_acc, on_step=True, on_epoch=False)
            self.log("classification_loss", class_loss)

            return class_loss

        else:
            reg_pred = self.regression(x_target.float())
            reg_loss = F.mse_loss(reg_pred, reg_target.float().view(-1, 1))
            self.log("regression_loss", reg_loss)
            return reg_loss

    def training_epoch_end(self, outs):
        # log epoch metric
        if self.mode == "classification":
            self.log("train_acc_epoch", self.train_acc.compute())
        else:
            self.regression = nn.Sequential(self.stem, self.fc3) #pass

    def validation_step(self, batch, batch_idx):
        x_target, class_target, reg_target, _ = batch
        if self.mode == "classification":
            class_pred = self.classification(x_target.float())
            class_loss = F.binary_cross_entropy_with_logits(
                class_pred, class_target.float()
            )
            self.valid_acc(torch.sigmoid(class_pred), class_target)
            self.log("valid_acc", self.valid_acc.compute())
            self.log("classification_loss", class_loss)
            return class_loss

        else:
            reg_pred = self.regression(x_target.float())
            reg_loss = F.mse_loss(reg_pred, reg_target.float().view(-1, 1))
            self.log("regression_loss", reg_loss)
            return reg_loss

    def configure_optimizers(self):
        optimizer  = torch.optim.SGD(self.parameters(), lr=0.01, weight_decay=0.00001, momentum=0.95)
        return optimizer

    def forward(self, x):
        if self.mode == "classification":
            class_pred = self.classification(x.float())
            return {"class": torch.sigmoid(class_pred)}
        else:
            reg_pred = self.regression(x.float())
            return {"energy": reg_pred}
import json
import os
import os.path
import math
import time
import logging
import configargparse

import torch
from torch.nn import DataParallel
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
#from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter

from models.linear import MeanPoolingLinear,OnlyLinear
from utils import to_device

from dataloader import create_loader_with_folds
from dataloader import _create_loader

from losses.mcc import MinimumClassConfusionLoss

# - To initialize the weights in each fold
def weight_reset(m):
    if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
        m.reset_parameters()

class Trainer:
    def __init__(self, params: configargparse.Namespace):
        """Initializes the Trainer with the training args provided in train.py"""

        logging.basicConfig(
            filename=os.path.join(params.log_dir, "train.log"),
            filemode="a",
            format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
            datefmt="%H:%M:%S",
            level=logging.INFO,
        )

        self.params = params
        self.nepochs = params.nepochs
        self.ngpu = params.ngpu
        
        # - loss functions
        self.loss = CrossEntropyLoss()
        self.mcc_loss = MinimumClassConfusionLoss(params.temperature)
        # - lambda of transfer loss(mcc_loss)
        self.transfer_loss_factor = params.transfer_loss_factor
        
        # - flag
        self.batch_enable = params.batch_enable
        self.handcrafted_features = params.handcrafted_features
        self.only_handcrafted_features = params.only_handcrafted_features
        self.fold = params.fold
        self.fold_prefix = "five"
        if self.fold == 10:
            self.fold_prefix = "ten"
        self.fold_id = -1
        self.mcc = params.mcc
        
        if self.batch_enable:
            print("batch_enable.")
            self.train_datasets = []
            self.valid_datasets = []
            self.test_datasets = []
            self.train_loaders = []
            self.valid_loaders = []
            self.test_loaders = []
            
            # - because batch sampler is created in _create_loader, 
            # - batch has different dataset creation method. In the
            # - end, we have to standarlize this interface
            for fold_id in range(self.fold):
                with open("inputdata/"+self.fold_prefix+"_fold_"+str(fold_id+1)+"_train.json", "rb") as f:
                    train_json = json.load(f)
                with open("inputdata/"+self.fold_prefix+"_fold_"+str(fold_id+1)+"_valid.json", "rb") as f:
                    valid_json = json.load(f)
                with open("inputdata/"+self.fold_prefix+"_fold_"+str(fold_id+1)+"_test.json", "rb") as f:
                    test_json = json.load(f)
                train_dataset, train_loader, _ =  _create_loader(
                    train_json,
                    params
                )
                valid_dataset, valid_loader, _ =  _create_loader(
                    valid_json,
                    params
                )
                test_dataset, test_loader, _ =  _create_loader(
                    test_json,
                    params
                )
                self.train_datasets.append(train_dataset)
                self.valid_datasets.append(valid_dataset)
                self.test_datasets.append(test_dataset)
                self.train_loaders.append(train_loader)
                self.valid_loaders.append(valid_loader)
                self.test_loaders.append(test_loader)
        else: # - TODO: sync with batch dataset creation
            with open(params.train_json, "rb") as f:
                train_json = json.load(f)
            self.train_datasets, self.train_loaders, self.valid_datasets, self.valid_loaders, self.test_datasets, self.test_loaders = create_loader_with_folds(train_json, params, is_train = True, is_aug = False)

        # - TODO: in some paper, they apply the pooling first then project.
        # - In this model, we apply the project->pooling->project method.
        # - the downside is that our padding in minibatch will impact the 
        # - accuracy. If we pooling first, we can pool the feature in dataloader
        # - and make things easy.
        if not self.only_handcrafted_features:
            self.model = MeanPoolingLinear(params.idim, params.odim, params.hidden_dim, self.handcrafted_features)
        else:
            # TODO: hidden_dim is int, 1 dimensional; ignore here
            # TODO: 9 is the number of handcrafted features
            self.model = OnlyLinear(9,params.odim)
        
        if torch.cuda.is_available():
            self.model = self.model.cuda()

        if self.ngpu > 1:
            self.model = DataParallel(self.model)

        logging.info(str(self.model))

        total_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        params.tparams = total_params
        logging.info(
            "Built a model with {:2.2f}M Params".format(float(total_params) / 1000000)
        )

        ## Write out model config
        with open(os.path.join(params.expdir, "model.json"), "wb") as f:
            f.write(json.dumps(vars(params), indent=4, sort_keys=True).encode("utf_8"))

        
        self.opt = Adam(
            self.model.parameters(), lr=params.lr, weight_decay=params.wdecay
        )
        # - TODO: add scheduler for better performance
        #self.scheduler = WarmupLR(self.opt, warmup_steps=params.warmup_steps)

        ## Initialize Stats for Logging
        self.train_stats = {}
        self.val_stats = {"best_acc": 0, "best_loss": 1e9, "best_epoch": -1}
        self.test_stats = {}
        self.writer = SummaryWriter(self.params.tb_dir)  # for tensorboard

        ## Resume/Load Model (TODO: this function is not tested.)
        if params.resume != "":
            self.resume_training(params.resume)
        else:
            self.epoch = 0
        self.start_time = time.time()

    # - To initialize the weights in each fold
    def model_init(self):
        logging.info(f"Model initialized")
        self.model.apply(weight_reset)

    def train(self):
        """Performs SER Training using the provided configuration.
        This is the main training wrapper that trains and evaluates the model across epochs
        """
        fold_acc = 0
        for fold_id in range(self.fold):
            self.fold_id = fold_id

            self.model_init()
            self.reset_statistic()
            
            self.train_sampler = self.train_loaders[fold_id]
            self.valid_sampler = self.valid_loaders[fold_id]
            self.test_sampler = self.test_loaders[fold_id]
            
            logging.info(f"Start to run {self.fold_prefix} fold {fold_id+1}/{self.fold}")
            self._train()
            fold_acc += self.test_stats["best_acc"]
        logging.info(f"{self.fold_prefix} fold acc: {fold_acc/self.fold}")

    def _train(self):
        while self.epoch < self.nepochs:
            self.reset_stats()
            start_time = time.time()

            logging.info(f"Start to train epoch {self.epoch}")
            self.train_epoch()

            logging.info(f"Start to validate epoch {self.epoch}")
            self.validate_epoch()
            
            logging.info(f"Start to test epoch {self.epoch}")
            self.test_epoch()
            
            end_time = time.time()
            ## Log Tensorboard and logfile
            log_str = (
                f"Epoch {self.epoch:02d}, lr={self.opt.param_groups[0]['lr']} | Train: acc={self.train_stats['acc']:.4f}"
                f" | Val: acc={self.val_stats['acc']:.4f} | Test: acc={self.test_stats['acc']:.4f} | "
                f"Time: this epoch {end_time - start_time:.2f}s, elapsed {end_time - self.start_time:.2f}s"
            )
            logging.info(log_str)
            self.log_epoch()

            self.save_model()
            self.epoch += 1
        logging.info(f"Best test acc={self.test_stats['best_acc']}")

    def train_epoch(self):
        """ "Contains the training loop across all training data to update the model in an epoch"""
        self.model.train()
        
        # TODO: fix batch
        for i, (feats, feat_len, handcrafted_features, target, key) in enumerate(
            self.train_sampler
        ):
            
            feats, feat_len,handcrafted_features, target = to_device(
                (feats, feat_len,handcrafted_features, target),
                next(self.model.parameters()).device,
            )

            y = self.model(feats, feat_len, handcrafted_features)
            
            train_acc = torch.sum(torch.argmax(y, axis=-1) == torch.argmax(target, axis=-1)).float()/len(target)
            loss = self.loss(y, target)
            loss /= self.params.accum_grad
            loss.backward()
            
            if (i + 1) % self.params.log_interval == 0:
                logging.info(
                    f"[Epoch {self.epoch}, Batch={i}] Train: loss={loss.item():.4f}, lr={self.opt.param_groups[0]['lr']}"
                )

            if (i + 1) % self.params.accum_grad == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.params.grad_clip
                )
                if math.isnan(grad_norm):
                    logging.info("[Warning] Grad norm is nan. Do not update model.")
                else:
                    self.opt.step()
                    # TODO: scheduler
                    #self.scheduler.step()

                self.opt.zero_grad()

            self.train_stats["nbatches"] += 1
            self.train_stats["loss"] += loss.item()
            self.train_stats["acc"] += train_acc
        
        '''
        In current setting, we consider test domain as 
        target domain here.
        '''
        if self.mcc: 
            for i, (feats, feat_len,handcrafted_features, target, key) in enumerate(
                self.test_sampler
            ):
                feats, feat_len, handcrafted_features, target = to_device(
                    (feats, feat_len,handcrafted_features, target),
                    next(self.model.parameters()).device,
                )
                
                y = self.model(feats, feat_len, handcrafted_features)
                loss = self.mcc_loss(y) * self.transfer_loss_factor
                loss /= self.params.accum_grad
                loss.backward()
                if (i + 1) % self.params.log_interval == 0:
                    logging.info(
                        f"[Epoch {self.epoch}, Batch={i}] Train(target): MCCloss={loss.item():.4f}, lr={self.opt.param_groups[0]['lr']}"
                    )

                if (i + 1) % self.params.accum_grad == 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.params.grad_clip
                    )
                    if math.isnan(grad_norm):
                        logging.info("[Warning] Grad norm is nan. Do not update model.")
                    else:
                        self.opt.step()
                        #self.scheduler.step() - TODO: scheduler

                    self.opt.zero_grad()
                
        
        self.train_stats["acc"] /= self.train_stats["nbatches"]
        self.train_stats["loss"] /= self.train_stats["nbatches"]

    def validate_epoch(self):
        """ "Contains the validation loop across all validation data in an epoch"""
        self.model.eval()

        with torch.no_grad():
            for i, (feats, feat_len,handcrafted_features, target, key) in enumerate(
                self.valid_sampler
            ):
                feats, feat_len, handcrafted_features, target = to_device(
                    (feats, feat_len, handcrafted_features, target),
                    next(self.model.parameters()).device,
                )

                y = self.model(feats, feat_len, handcrafted_features)
                loss = self.loss(y, target)
                val_acc = torch.sum(torch.argmax(y, axis = -1) == torch.argmax(target, axis = -1)).float()/len(target)

                self.val_stats["nbatches"] += 1
                self.val_stats["loss"] += loss.item()
                self.val_stats["acc"] += val_acc

            self.val_stats["acc"] /= self.val_stats["nbatches"]
            self.val_stats["loss"] /= self.val_stats["nbatches"]
            
    def test_epoch(self):
        """ "Contains the validation loop across all validation data in an epoch"""
        self.model.eval()

        with torch.no_grad():
            for i, (feats, feat_len,handcrafted_features, target, key) in enumerate(
                self.test_sampler
            ):
                feats, feat_len,handcrafted_features, target = to_device(
                    (feats, feat_len,handcrafted_features, target),
                    next(self.model.parameters()).device,
                )

                y = self.model(feats, feat_len, handcrafted_features)
                loss = self.loss(y, target)
                test_acc = torch.sum(torch.argmax(y, axis = -1) == torch.argmax(target, axis = -1)).float()/len(target)

                self.test_stats["nbatches"] += 1
                self.test_stats["loss"] += loss.item()
                self.test_stats["acc"] += test_acc

            self.test_stats["acc"] /= self.test_stats["nbatches"]
            if self.test_stats["best_acc"] < self.test_stats["acc"]:
                self.test_stats["best_acc"] = self.test_stats["acc"]
            self.test_stats["loss"] /= self.test_stats["nbatches"]
        
    def resume_training(self, path: str):
        """
        Utility function to load a previous model and optimizer checkpoint, and set the starting epoch for resuming training
        """
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.opt.load_state_dict(checkpoint["optimizer_state_dict"])
        self.epoch = checkpoint["epoch"] + 1
        self.val_stats["best_epoch"] = checkpoint["epoch"]
        self.val_stats["best_loss"] = checkpoint["loss"]
        self.val_stats["best_acc"] = checkpoint["acc"]

    def reset_statistic(self):
        self.epoch = 0
        self.train_stats = {}
        self.val_stats = {"best_acc": 0, "best_loss": 1e9, "best_epoch": -1}
        self.test_stats = {"best_acc": 0}
        
    def reset_stats(self):
        """
        Utility function to reset training and validation statistics at the start of each epoch
        """
        self.train_stats["nbatches"] = 0
        self.train_stats["loss"] = 0
        self.train_stats["acc"] = 0

        self.val_stats["nbatches"] = 0
        self.val_stats["loss"] = 0
        self.val_stats["acc"] = 0
        
        self.test_stats["nbatches"] = 0
        self.test_stats["loss"] = 0
        self.test_stats["acc"] = 0

    def save_model(self):
        """Save the model snapshot after every epoch of training."""
        if self.val_stats["acc"] > self.val_stats["best_acc"]:
            old_ckpt = os.path.join(
                self.params.model_dir, f'epoch{self.val_stats["best_epoch"]}.pth'
            )
            if os.path.exists(old_ckpt):
                os.remove(old_ckpt)
            self.val_stats["best_epoch"] = self.epoch
            self.val_stats["best_acc"] = self.val_stats["acc"]

            torch.save(
                {
                    "epoch": self.epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.opt.state_dict(),
                    "loss": self.val_stats["loss"],
                    "acc": self.val_stats["acc"],
                },
                os.path.join(self.params.model_dir, f"epoch{self.epoch}.pth"),
            )
            logging.info(f"[info] Save model after epoch {self.epoch}\n")

    def log_epoch(self):
        """Write stats from the Training and Validation Statistics Dictionaries onto Tensorboard at the end of each epoch"""
        self.writer.add_scalar("training/"+str(self.fold_id)+"/acc", self.train_stats["acc"], self.epoch)
        self.writer.add_scalar("validation/"+str(self.fold_id)+"/acc", self.val_stats["acc"], self.epoch)
        self.writer.add_scalar("test/"+str(self.fold_id)+"/acc", self.test_stats["acc"], self.epoch)

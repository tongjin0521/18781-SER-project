# 
import sys
import os
import configargparse
import random
import torch
import numpy as np
configargparse
from trainer import Trainer


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def get_parser(parser=None, required=True):
    if parser is None:
        parser = configargparse.ArgumentParser(
            description="Train an speech emotion recognition (SER) model",
            config_file_parser_class=configargparse.YAMLConfigFileParser,
            formatter_class=configargparse.ArgumentDefaultsHelpFormatter,
        )

    parser.add(
        "--config",
        is_config_file=True,
        help="config file path",
        default="conf/base.yaml",
    )

    ## General utils
    parser.add_argument(
        "--tag", type=str, help="Experiment Tag for storing logs, models"
    )
    parser.add_argument("--seed", default=2022, type=int, help="Random seed(default 2022)")
    parser.add_argument(
        "--text_pad", default=-1, type=int, help="Padding Index for Text Labels"
    )
    parser.add_argument(
        "--audio_pad", default=0, type=int, help="Padding Index for Audio features"
    )
    parser.add_argument(
        "--mcc", default=False, type=bool, help="Activate mimimum class confusion loss"
    )
    parser.add_argument(
        "--transfer_loss_factor", default=1.0, type=float, help="weight of transfer_loss"
    )
    parser.add_argument(
        "--batch_size", default=1, type=int, help="batch_size"
    )
    parser.add_argument(
        "--temperature", default=2.0, type=float, help="temperature for mcc loss"
    )
    parser.add_argument(
        "--pad", default=0, type=int, help="Padding Index for Audio features"
    )
    parser.add_argument(
        "--handcrafted_features", default=False, type=bool, help="Include handcrafted features"
    )
    parser.add_argument(
        "--mfcc_included", default=False, type=bool, help="Include mfcc features"
    )
    parser.add_argument(
        "--only_handcrafted_features", default=False, type=bool, help="Train&test on only handcrafted features"
    )
    parser.add_argument(
        "--target_pad", default=0, type=int, help="???"
    )
    parser.add_argument(
        "--batch_enable", default=False, type=bool, help="This option impact how the dataloader interact with models"
    )
    parser.add_argument(
        "--fold", default=5, type=int, help="10 fold or 5 fold"
    )
    parser.add_argument(
        "--final_dropout", default=0.1, type=float, help="pretrain last layer drop out"
    )
    parser.add_argument(
        "--vocab_size", default=32, type=int, help="voacbulary size for ASR in MTL"
    )
    
    ## I/O related
    parser.add_argument(
        "--train_json",
        type=str,
        default="inputdata/sample.json",
        help="Filename of train label data (json)",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=".",
        help="Output Data Directory/Experiment Directory",
    )

    ## Encoder related
    parser.add_argument(
        "--idim", type=int, default=768, help="Input Feature Size"
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=256,
        help="Hidden size of down stream model.",
    )
    parser.add_argument(
        "--odim", type=int, default=4, help="Output Feature Size"
    )

    ## Batch related
    parser.add_argument(
        "--batch_bins",
        type=int,
        default=800000,
    )
    parser.add_argument(
        "--nworkers",
        dest="nworkers",
        type=int,
        default=1,
    )

    ## Optimization related
    parser.add_argument("--lr", type=float, default=1e-4) # - s3prl use 1.0e-4
    parser.add_argument("--grad_clip", type=float, default=1) # - s3prl use 1.0
    parser.add_argument("--wdecay", type=float, default=0, help="Weight decay")
    parser.add_argument(
        "--accum_grad", default=8, type=int, help="Number of gradient accumuration" # - s3prl use 8
    )
    parser.add_argument(
        "--warmup_steps", default=25000, type=int, help="Number of lr warmup steps."
    )

    ## Training config
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="Path of model and optimizer to be loaded",
    )
    
    parser.add_argument(
        "--nepochs", type=int, default=50, help="Number of training epochs" # - s3prl use 30000
    ) 
    parser.add_argument(
        "--ngpu",
        default=1,
        type=int,
        help="Number of GPUs. If not given, use all visible devices",
    )
    parser.add_argument(
        "--log_interval", default=200, type=int, help="Log interval in batches."
    )

    return parser


def main(cmd_args):
    ## Return the arguments from parser
    parser = get_parser()
    args, _ = parser.parse_known_args(cmd_args)
    
    if args.fold != 5 and args.fold != 10:
        print(f'fold must be 5 or 10, it is {args.fold}')
        return
    
    ## Set Random Seed for Reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    ## Set directories
    expdir = os.path.join(args.out_dir, "exp", "train_" + args.tag)
    model_dir = os.path.join(expdir, "ckpts")
    log_dir = os.path.join(expdir, "logs")
    tb_dir = os.path.join(expdir, "tensorboard")

    args.expdir = expdir
    args.model_dir = model_dir
    args.log_dir = log_dir
    args.tb_dir = tb_dir

    for x in [expdir, model_dir, log_dir, tb_dir]:
        os.makedirs(x, exist_ok=True)

    ## Start training
    trainer = Trainer(args)
    trainer.train()


if __name__ == "__main__":
    main(sys.argv[1:])

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
import yaml
import pickle
import argparse
import warnings
import torch
from libs.utils import *
from libs.train import *
from omegaconf import OmegaConf
from sklearn.model_selection import KFold, train_test_split
warnings.filterwarnings('ignore')

def main(args):

    # --------------------------------------------------------------------------
    # loads configs
    with open(args.config_file, 'r') as f:
        configs = yaml.load(f, Loader=yaml.Loader)
    configs = OmegaConf.create(configs)

    # --------------------------------------------------------------------------
    # loads data splits and stats    
    df: pd.DataFrame = pd.read_csv(args.train_df)
    train_df, dev_df = train_test_split(df, test_size=0.2, random_state=42)

    train_list = train_df['fname'].tolist()
    val_list = dev_df['fname'].tolist()
    print('train and val data', len(train_list), len(val_list))

    # --------------------------------------------------------------------------
    # initialize environments
    init_environment(configs.seed)

    print('-' * 100)
    print('Training ...\n')
    print(f'- Data Root : {args.data_root}')
    print(f'- Exp Dir   : {args.exp_root}')
    print(f'- Config file   : {args.config_file}')
    print(f'- Configs   : {args.config_file}')
    print(f'- Normalization : {args.norm_label}')
    print(f'- Num Train : {len(train_list)}')
    print(f'- Num Val   : {len(val_list)}\n')

    loader_kwargs = dict(
        configs        = configs.loader#,
    )
    train_loader = get_dataloader('train', train_list, args.data_root, args.norm_label, **loader_kwargs)
    val_loader   = get_dataloader('val',   val_list,   args.data_root, args.norm_label, **loader_kwargs)    
    print('torch.cuda.mem_get_info()', torch.cuda.mem_get_info())

    # initialize trainer
    trainer = PytorchTrainer(
        configs = configs,
        exp_dir = args.exp_root,
        resume  = args.resume,
        label_norm = args.norm_label
    )

    # training model
    trainer.forward(train_loader, val_loader)    
    print('-' * 100, '\n')    
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument( "--train-df", type=str, default='<PATH TO TRAIN CSV LIST>', help="path to train csv datalist",)
    parser.add_argument('--data_root',      type=str, default='<PATH TO TRAINNING DATA>', help='dir path of training data')
    parser.add_argument('--exp_root',       type=str, default='<PATH TO CKPT>', help='root dir of experiments')
    parser.add_argument('--config_file',    type=str, default ='<CONFIG YAML FILE>', help='yaml path of configs')
    parser.add_argument('--resume',         action='store_true', help='if resume from checkpoint')
    parser.add_argument('--norm_label',     action='store_true', help='if use normalized labels')
    args = parser.parse_args()

    check_train_args(args)
    main(args)
import yaml
import pickle
import argparse
import warnings
from libs.utils import *
from libs.predict import *
from omegaconf import OmegaConf
from os.path import join as opj
from sklearn.model_selection import KFold, train_test_split
warnings.filterwarnings('ignore')

def main(args):
    # --------------------------------------------------------------------------
    # loads configs
    with open(args.config_file, 'r') as f:
        configs = yaml.load(f, Loader=yaml.Loader)
    configs = OmegaConf.create(configs)

    # loads data splits and stats    
    df: pd.DataFrame = pd.read_csv(args.test_df)
        
    if args.val:
        train_df, dev_df = train_test_split(df, test_size=0.2, random_state=42)
        val_list = dev_df['fname'].tolist()
    
    else:
        val_list = df['fname'].tolist()
    print('Test data size', len(val_list))

    # --------------------------------------------------------------------------
    # predicting test data
    exp_dir = opj(args.exp_root)
    data_dir = args.data_root
    output_dir = opj(args.output_root)

    print('-' * 100)
    print('Predicting ...\n')
    print(f'- Data Dir  : {data_dir}')
    print(f'- Exp Dir   : {exp_dir}')
    print(f'- Out Dir   : {output_dir}')
    print(f'- Configs   : {args.config_file}')

    model_paths = opj(exp_dir, 'model.pth')
    predictor = BHEPredictor(
        model_path    = model_paths,
        configs        = configs,
    )
    predictor.predict(data_dir, val_list, output_dir)
    
    print('-' * 100, '\n')
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predicting')
    parser.add_argument( "--test-df", type=str, default='<PATH TO TEST CSV LIST>', help="path to test csv datalist",)
    parser.add_argument('--data_root',      type=str, default='<PATH TO TEST DATA>', help='dir path of test data')
    parser.add_argument('--exp_root',       type=str, default='<EXPERIMENT CKPT FOLDER>', help='root dir of experiments')
    parser.add_argument('--output_root',    type=str, default='<OUTPUT PREDICTION FOLDER>', help='root dir of outputs')
    parser.add_argument('--config_file',    type=str, default ='<CONFIG YAML>', help='yaml path of configs')
    parser.add_argument('--val',     action='store_true', help='if validatin accuracy else test accuracy')
    args = parser.parse_args()
    check_predict_args(args)
    main(args)
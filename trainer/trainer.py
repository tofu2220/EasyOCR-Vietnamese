import os

import pandas as pd
import torch.backends.cudnn as cudnn
import yaml
from train import train
from utils import AttrDict
from model import Model

cudnn.benchmark = True
cudnn.deterministic = False


def get_config(file_path):
    with open(file_path, 'r', encoding="utf8") as stream:
        opt = yaml.safe_load(stream)
    opt = AttrDict(opt)
    if opt.lang_char == 'None':
        characters = ''
        for data in opt['select_data'].split('-'):
            csv_path = os.path.join(opt['train_data'], data, 'labels.csv')
            df = pd.read_csv(csv_path, sep=',', usecols=['filename', 'words'], keep_default_na=False)
            all_char = ''.join(df['words'])
            characters += ''.join(set(all_char))
        characters = sorted(set(characters))
        opt.character = ''.join(characters)
    else:
        opt.character = opt.number + opt.symbol + opt.lang_char
    os.makedirs(f'./saved_models/{opt.experiment_name}', exist_ok=True)
    return opt


if __name__ == "__main__":
    # Load the config
    # opt = get_config("config_files/en_filtered_config.yaml")
    opt = get_config("config_files/vi_filtered_config.yaml")
    print(f"opt.character: {opt.character}")
    
    # Construct csv_path manually for testing
    train_data = opt['train_data']  # 'all_data'
    select_data = opt['select_data']  # 'en_train_filtered'
    for data in select_data.split('-'):  # Split by '-' and iterate
        csv_path = os.path.join(train_data, data, 'labels.csv')
        print(f"csv_path: {csv_path}")
        
        # Test if the file exists
        if os.path.exists(csv_path):
            print(f"File exists at: {csv_path}")
            # Optionally, read and print the first few rows to verify
            df = pd.read_csv(csv_path, sep=',', usecols=['filename', 'words'], keep_default_na=False)
            all_chars = ''.join(df['words'])
            unique_chars = sorted(set(all_chars))
            missing_chars = [c for c in unique_chars if c not in opt.character]
            print(f"Characters in labels: {unique_chars}")
            print(f"Missing characters (not in opt.character): {missing_chars}")
            print("First few rows of the CSV:")
            print(df.head())
        else:
            print(f"File does not exist at: {csv_path}")

    print("Training started...")
    train(opt, amp=False)
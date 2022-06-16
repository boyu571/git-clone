from datetime import datetime, timezone, timedelta
from imp import load_dynamic
from sched import scheduler
import numpy as np
import random
import os
import copy
import torch
import torchvision
from modules.datasets import *
from modules.utils import *
from modules.optimizers import get_optimizer
from modules.scheduler import PolyLR
from models.utils import get_model, EMA


import wandb
import warnings
#from baseline.train import PROJECT_DIR
warnings.filterwarnings('ignore')

#from modules


# Root 디렉토리
PROJECT_DIR = os.path.dirname(__file__)

print(PROJECT_DIR)

# config 정보 로드
config_path = os.path.join(PROJECT_DIR, 'config', 'train_config.yml')
config = load_yaml(config_path)

# pre-trained model 
pre_trained = config['MODEL']['pre_model1'] # 1: reco, 2: cps, 3: ps-mt
pre_models = ['reco', 'cps', 'ps-mt']

# Train Serial
kst = timezone(timedelta(hours=9))
train_serial = datetime.now(tz=kst).strftime("%Y%m%d_%H%M%S")

# 기록 디렉토리
RECORDER_DIR = os.path.join(PROJECT_DIR, 'results', 'train', train_serial)
os.makedirs(RECORDER_DIR, exist_ok=True)

# 데이터 디렉토리
DATA_DIR = os.path.join(PROJECT_DIR, 'data', config['DIRECTORY']['dataset'])

# Seed
torch.manual_seed(config['TRAINER']['seed'])
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(config['TRAINDER']['seed'])
random.seed(config['TRAINER']['seed'])

# GPU
#os.environ['CUDA_VISIBLE_DEVICES'] = str(config['TRAINER']['gpu'])
os.environ['CUDA_VISIBLE_DEVICES'] = 2, 3
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
    """
    00. Set Logger
    """
    logger = get_logger(name='train', dir_=RECORDER_DIR, stream=False)
    logger.info(f"Set Logger {RECORDER_DIR}")

    """
    01. Load data
    """
    # Dataset
    data_loader = BuildDataLoader()
    train_l_loader, train_u_loader, valid_l_loader, _ = data_loader.build(supervised=False)
    logger.info(f'Load data, train (labeled):{len(train_l_loader)} train (unlabeled):{len(train_u_loader)} val:{len(valid_l_loader)}')

    """
    02. set model
    """
    model = get_model(model_name='deeplabv3p', num_classes=5, output_dim=pre_trained['output_dim'].to(device))
    ema = EMA(model, 0.99)

    """
    3. Set trainer
    """
    # Optimizer
    optimizer_reco = get_optimizer(optimizer_name=config['TRAINER']['optimizer'])
    optimizer_reco = optimizer_reco(params=model.parameters(),lr=config['TRAINER']['learning_rate'])
    scheduler_reco = PolyLR(optimizer_reco,config['TRAINER']['n_epochs'])


    # Early stopper
    early_stopper = early_stopper()

    # Trainer
    trainer = Trainer(model = model,
                    ema = ema,
                    data_loader = data_loader,
                    optimizer= optimizer
                    device=device, # cuda
                    logger = logger,
                    config = config['TRAINER'],
                    interval=config['LOGGER']['logging_interval']
                    )
    """
    logger
    """
    # Recorder
    recorder = 

    # Wandb
    if config['LOGGER']['wandb'] == True: #본인 wandb 계정입력
        wandb_project_serial = f'{pre_trained}_'
        wandb_username = 'a22106'

    # Save train config
    save_yaml(os.path.join(RECORDER_DIR, 'train_config.yml'), config)

    """
    4. Train
    """

    # Train
    n_epochs = config['TRAINER']['n_epochs']
    for epoch_index in range(n_epochs):
        # Set Recorder row
        row_dict = dict()
        row_dict['epoch_index'] = epoch_index
        row_dict['train_serial'] = train_serial

        """
        Train
        """
        print(f"Train {epoch_index}/{n_epochs}")
        logger.info(f"--Train {epoch_index}/{n_epochs}")
        trainer.train(train_l_loader=train_l_loader, train_u_loader=train_u_loader)

        row_dict['train_loss'] = trainer.loss_mean
        row_dict['train_elapsed_time'] = trainer.elapsed_time 
        
        for metric_str, score in trainer.score_dict.items():
            row_dict[f"train_{metric_str}"] = score
        trainer.clear_history()
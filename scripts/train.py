import argparse
import os
import pickle
import sys
import yaml

sys.path.append(".")

from torch.utils.data import DataLoader

from src.model import *
from src.trainer import *
from src.dataset import MidiDataset

# General settings
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='conf.yml')
parser.add_argument('--model_type', type=str, default='lstm')
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--resume', type=bool, default=False)


def load_model(model_type, params):
    if model_type == 'lstm':
        model = MusicLSTMVAE(**params)
    elif model_type == 'gru':
        model = MusicGRUVAE(**params)
    else:
        raise Exception("Invalid model type. Expected lstm or gru")
    return model


def load_data(train_data, val_data, batch_size, validation_split=0.2, random_seed=874):
    train_loader = None
    val_loader = None
    if train_data != '':
        X_train = pickle.load(open(train_data, 'rb'))
        train_data = MidiDataset(X_train)
        train_loader = DataLoader(train_data, batch_size=batch_size)
    if val_data != '':
        X_val = pickle.load(open(val_data, 'rb'))
        val_data = MidiDataset(X_val)
        val_loader = DataLoader(val_data, batch_size=batch_size)
    
    return train_loader, val_loader


def train(model, trainer, train_data, val_data, epochs, resume):
    """
    Train a model
    """
    trainer.train(model, train_data, None, epochs, resume, val_data)


def main(args):
    model_params = None
    trainer_params = None
    data_params = None
    with open(args.config, 'r') as config_file:
        config = yaml.load(config_file)
        model_params = config['model']
        trainer_params = config['trainer']
        data_params = config['data']
        
    train_data, val_data = load_data(data_params['train_data'], 
                                     data_params['val_data'], 
                                     trainer_params['batch_size'])
    
    model = load_model(args.model_type, model_params)

    trainer = Trainer(**trainer_params)

    train(model, trainer, train_data, val_data, args.epochs, args.resume)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)

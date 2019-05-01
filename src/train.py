import argparse
import os
import pickle
import sys

sys.path.append(".")

from torch.utils.data import DataLoader

from src.model import *
from src.trainer import *
from src.dataset import MidiDataset
from src.transform import Transform

# General settings
parser = argparse.ArgumentParser()
parser.add_argument('--train_data', default='data/train.pickle')
parser.add_argument('--val_data', default='data/val.pickle')
parser.add_argument('--model_type', type=str, default='lstm')
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--resume', type=bool, default=False)

# Model parameters
parser.add_argument('--num_subsequences', type=int, default=16)
parser.add_argument('--max_sequence_length', type=int, default=256)
parser.add_argument('--sequence_length', type=int, default=16)
parser.add_argument('--encoder_input_size', type=int, default=61)
parser.add_argument('--decoder_input_size', type=int, default=61)
parser.add_argument('--encoder_hidden_size', type=int, default=2048)
parser.add_argument('--decoder_hidden_size', type=int, default=1024)
parser.add_argument('--latent_dim', type=int, default=512)
parser.add_argument('--encoder_num_layers', type=int, default=2)
parser.add_argument('--decoder_num_layers', type=int, default=2)

# Trainer parameters
parser.add_argument('--learning_rate', type=float, default=1e-3)
parser.add_argument('--KL_rate', type=float, default=0.9999)
parser.add_argument('--free_bits', type=int, default=256)
parser.add_argument('--sampling_rate', type=int, default=2000)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--print_every', type=int, default=1000)
parser.add_argument('--checkpoint_every', type=int, default=10000)
parser.add_argument('--checkpoint_dir', type=str, default='checkpoint')
parser.add_argument('--output_dir', type=str, default='outputs')
#parser.add_argument('--num_workers', type=int, default=1)


def load_model(model_type, params):
    if model_type == 'lstm':
        model = MusicLSTMVAE(**params)
    elif model_type == 'gru':
        model = MusicGRUVAE(**params)
    else:
        raise Exception("Invalid model type. Expected lstm or gru")
    return model


def load_data(train_data, val_data, batch_size, validation_split=0.2, random_seed=874):
    X_train = pickle.load(open(train_data, 'rb'))
    X_val = pickle.load(open(val_data, 'rb'))
    train_data = MidiDataset(X_train, Transform(1))
    val_data = MidiDataset(X_val, Transform(1))
    train_loader = DataLoader(train_data, batch_size=batch_size)
    val_loader = DataLoader(val_data, batch_size=batch_size)
    return train_loader, val_loader


def train(model, trainer, train_data, val_data, epochs, resume):
    """
    Train a model
    """
    trainer.train(model, train_data, None, epochs, resume, val_data)


def main(args):
    train_data, val_data = load_data(args.train_data, args.val_data, args.batch_size)
    model_params = {
        'num_subsequences': args.num_subsequences,
        'max_sequence_length': args.max_sequence_length,
        'sequence_length': args.sequence_length,
        'encoder_input_size': args.encoder_input_size,
        'decoder_input_size': args.decoder_input_size,
        'encoder_hidden_size': args.encoder_hidden_size,
        'decoder_hidden_size': args.decoder_hidden_size,
        'latent_dim': args.latent_dim,
        'encoder_num_layers': args.encoder_num_layers,
        'decoder_num_layers': args.decoder_num_layers
    }
    model = load_model(args.model_type, model_params)

    trainer_params = {
        'learning_rate': args.learning_rate,
        'KL_rate': args.KL_rate,
        'free_bits': args.free_bits,
        'sampling_rate': args.sampling_rate,
        'batch_size': args.batch_size,
        'print_every': args.print_every,
        'checkpoint_every': args.checkpoint_every,
        'checkpoint_dir': args.checkpoint_dir,
        'output_dir': args.output_dir
    }

    trainer = Trainer(**trainer_params)

    train(model, trainer, train_data, val_data, args.epochs, args.resume)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)

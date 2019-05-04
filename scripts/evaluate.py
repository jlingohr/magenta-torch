import argparse
import os
import pickle
import sys
import yaml

sys.path.append(".")

from torch.utils.data import DataLoader

from src.model import *
from src.sampler import *
from src.dataset import MidiDataset
from src.midi_functions import rolls_to_midi

# General settings
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='conf.yml')
parser.add_argument('--model_type', type=str, default='lstm')
parser.add_argument('--mode', type=str, choices=['eval', 'interpolate', 'reconstruct'])


def load_model(model_type, params):
    if model_type == 'lstm':
        model = MusicLSTMVAE(**params)
    elif model_type == 'gru':
        model = MusicGRUVAE(**params)
    else:
        raise Exception("Invalid model type. Expected lstm or gru")
    return model

def load_data(test_data, batch_size, song_paths='', instrument_path='', tempo_path=''):
    X_test = pickle.load(open(test_data, 'rb'))
    
    song_names = None
    if song_paths != '':
        song_names = [os.path.basename(x) for x in pickle.load(open(song_paths, 'rb'))]
    
    instruments = None
    if instrument_path != '':
        instruments = pickle.load(open(instrument_path, 'rb'))
    
    tempos = None
    if tempo_path != '':
        tempos = pickle.load(open(tempo_path, 'rb'))
    
    test_data = MidiDataset(X_test, song_paths=song_names, instruments=instruments, tempos=tempos)
    test_loader = DataLoader(test_data, batch_size=batch_size)
    return test_loader

def load_tempo(tempo_path, song_id):
    if temp_path is None:
        raise ValueError('Tempo file unspecified')
    else:
        tempos = pickle.load(open(tempo_path, 'rb'))
        return tempos[song_id]
    
def evaluate(sampler, model, args):
    data_path = args['test_data']
    song_names = args['test_songs']
    batch_size = args['batch_size']
    data = load_data(test_data=data_path, batch_size=batch_size, instrument_path='', song_paths=song_names)
    loss_tf, loss = sampler.evaluate(model, data)
    print("Loss with teacher forcing: %.4f, loss without teacher forcing: %.4f" % (loss_tf, loss))
    
def instrument_representation_to_programs(I, instrument_attach_method='1hot-category'):
    programs = []
    for instrument_vector in I:
        if instrument_attach_method == '1hot-category':
            index = np.argmax(instrument_vector)
            programs.append(index * 8)
        elif instrument_attach_method == 'khot-category':
            nz = np.nonzero(instrument_vector)[0]
            index = 0
            for exponent in nz:
                index += 2^exponent
            programs.append(index * 8)
        elif instrument_attach_method == '1hot-instrument':
            index = np.argmax(instrument_vector)
            programs.append(index)
        elif instrument_attach_method == 'khot-instrument':
            nz = np.nonzero(instrument_vector)[0]
            index = 0
            for exponent in nz:
                index += 2^exponent
            programs.append(index)
    return programs
    
def reconstruct(sampler, model, evaluation_params):
    # Load data
    data_path = evaluation_params['test_data']
    song_names = evaluation_params['test_songs']
    tempos = evaluation_params['test_tempos']
    instruments = evaluation_params['test_instruments']
    batch_size = evaluation_params['batch_size']
    data = load_data(data_path, batch_size, song_names, instruments, tempos)
    
    # Reconstruct specified song
    reconstruction_params = evaluation_params['reconstruction']
    song_id = reconstruction_params['song_name']
    temperature = evaluation_params['temperature']
    attach_method = reconstruction_params['attach_method']
    reconstruction_path = reconstruction_params['reconstruction_path']
    song = data.dataset.get_tensor_by_name(song_id)
    # Generate reconstruction from the samples
    reconstructed = sampler.reconstruct(model, song, temperature)
    # Reconstruct into midi form
    I, tempo = data.dataset.get_aux_by_names(song_id)
    programs = instrument_representation_to_programs(I, attach_method)
    
    rolls_to_midi(reconstructed, 
                  programs, 
                  reconstruction_path, 
                  song_id, 
                  tempo, 
                  24,
                  84,
                  128,
                  0.5)
    
    print('Saved reconstruction for %s' % song_id)
    
def interpolate():
    raise ValueError("Not implemented")

def main(args):
    model_params = None
    sampler = None
    data_params = None
    evaluation_params = None
    with open(args.config, 'r') as config_file:
        config = yaml.load(config_file)
        model_params = config['model']
        sampler_params = {
            'free_bits': config['sampler']['free_bits'],
            'output_dir': config['sampler']['output_dir']
        }
        data_params = config['data']     
        evaluation_params = config['evaluation']

    model = load_model(args.model_type, model_params)
    sampler = Sampler(**sampler_params)
    
    model.load_state_dict(torch.load(evaluation_params['model_path'], 
                                     map_location='cpu').state_dict(), strict=False)
    print(model)
    model.eval()
            
    mode = args.mode
    if mode == 'eval':
        evaluate(sampler, model, evaluation_params)
    elif mode == 'reconstruct':
        reconstruct(sampler, model, evaluation_params)
        
#     elif mode == 'interpolate':
#         song_id_A = args.song_id_a
#         song_id_B = args.song_id_b
#         data = load_data(data_path, args.batch_size)
#         song_a = data.get_tensor_by_name(song_id_a)
#         song_b = data.get_tensor_by_name(song_id_b)
#         interpolated = sampler.interpolate(model, song_a, song_b)
#         # TODO save interpolated

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)

import argparse
import os
import pickle
import sys
import yaml

sys.path.append(".")

from src.preprocess import MidiPreprocessor

# General settings
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='conf.yml')
parser.add_argument('--import_dir', type=str)
parser.add_argument('--save_imported_midi_as_pickle', type=bool, default=True)
parser.add_argument('--save_preprocessed_midi', type=bool, default=True)

def main(args):
    conf = None
    
    with open(args.config, 'r') as config_file:
        config = yaml.load(config_file)
        conf = config['preprocessor']
    
    processor = MidiPreprocessor(**conf)
    processor.import_midi_from_folder(args.import_dir,
                                     args.save_imported_midi_as_pickle,
                                     args.save_preprocessed_midi)
    
if __name__ == '__main__':
    args = parser.parse_args()
    main(args)

import torch
from torch.utils.data import Dataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MidiDataset(Dataset):
    """
    Inputs:
    - song_list: input pitches of shape (num_samples, input_length, different_pitches)
    - output_pitches: output pitches of shape (num_samples, output_length, different_pitches)
    """
    def __init__(self, song_list, transform, song_names=None, midi_path=None):
        self.song_list = [x.astype(float) for x in song_list]
        self.song_names = song_names
        self.transform = transform
        self.index_mapper, self.song_to_idx = self._initialize()
        self.midi_path = midi_path
    
    def _initialize(self):
        index_mapper = []
        song_to_idx = dict()
        
        for song_idx in range(len(self.song_list)):
            song_tuples = []
            
            split_count = self.transform.get_sections(self.song_list[song_idx].shape[0])
            for bar_idx in range(0, split_count):
                song_tuples.append((song_idx, bar_idx))
                index_mapper.append((song_idx, bar_idx))
                
            if self.song_names is not None:
                song_name = self.song_names[song_idx]
                song_to_idx[song_name] = song_tuples
                
        return index_mapper, song_to_idx
        
    def __len__(self):
        return len(self.index_mapper)
        
    def __getitem__(self, idx):
        """
        A sample is B consecutive bars. In MusicVAE this would be 16 consecutive
        bars. For MidiVAE a sample consists of a single bar. 
        """
        song_idx, section_idx = self.index_mapper[idx]
        
        sample = self.song_list[song_idx]
        sample = self.transform(sample, section_idx)
        x = torch.tensor(sample, dtype=torch.float).view(-1, 61)
        return x.to(device) 
    
    def get_song_by_name(self, song_name):
        indices = self.song_to_idx[song_name]
        samples = [self.song_list[song_idx] for song_idx in indices]
        print(samples.shape)
        return samples
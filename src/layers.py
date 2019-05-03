import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BiLSTMEncoder(nn.Module):
    """
    Bi-directional LSTM encoder from MusicVAE
    Inputs:
    - input_size: Dimension of one-hot representation of input notes
    - hidden_size: hidden size of bidirectional lstm
    - num_layers: Number of layers for bidirectional lstm
    """
    def __init__(self,
                 input_size=61,
                 hidden_size=2048,
                 latent_size=512,
                 num_layers=2):
        super(BiLSTMEncoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.num_layers = num_layers

        self.bilstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, bidirectional=True)
        self.mu = nn.Linear(in_features=2 * hidden_size, out_features=latent_size)
        self.sigma = nn.Linear(in_features=2 * hidden_size, out_features=latent_size)
        self.softplus = nn.Softplus()

    def forward(self, input, h0, c0):
        batch_size = input.size(1)
        _, (h_n, c_n) = self.bilstm(input, (h0, c0))
        h_n = h_n.view(self.num_layers, 2, batch_size, -1)[-1].view(batch_size, -1)
        mu = self.mu(h_n)
        sigma = self.softplus(self.sigma(h_n))
        return mu, sigma

    def init_hidden(self, batch_size=1):
        # Bidirectional lstm so num_layers*2
        return (torch.zeros(self.num_layers * 2, batch_size, self.hidden_size, dtype=torch.float, device=device),
                torch.zeros(self.num_layers * 2, batch_size, self.hidden_size, dtype=torch.float, device=device))


class HierarchicalLSTMDecoder(nn.Module):
    """
    Hierarchical decoder from MusicVAE
    """

    def __init__(self,
                 num_embeddings,
                 input_size=61,
                 hidden_size=1024,
                 latent_size=512,
                 num_layers=2,
                 max_seq_length=256,
                 seq_length=16):
        super(HierarchicalLSTMDecoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.num_embeddings = num_embeddings
        self.max_seq_length = max_seq_length
        self.seq_length = seq_length
        self.num_layers = num_layers

        self.tanh = nn.Tanh()
        self.conductor = nn.LSTM(input_size=latent_size, hidden_size=hidden_size, num_layers=num_layers)
        self.conductor_embeddings = nn.Sequential(
            nn.Linear(in_features=hidden_size, out_features=latent_size),
            nn.Tanh())
        self.lstm = nn.LSTM(input_size=input_size + latent_size, hidden_size=hidden_size, num_layers=num_layers)
        self.out = nn.Sequential(
            nn.Linear(in_features=hidden_size, out_features=input_size),
            nn.Softmax(dim=2)
        )

    def forward(self, target, latent, h0, c0, use_teacher_forcing=True, temperature=1.0):
        batch_size = target.size(1)

        out = torch.zeros(self.max_seq_length, batch_size, self.input_size, dtype=torch.float, device=device)
        # Initialie start note
        prev_note = torch.zeros(1, batch_size, self.input_size, dtype=torch.float, device=device)

        # Conductor produces an embedding vector for each subsequence
        for embedding_idx in range(self.num_embeddings):
            embedding, (h0, c0) = self.conductor(latent.unsqueeze(0), (h0, c0))
            embedding = self.conductor_embeddings(embedding)

            # Initialize lower decoder hidden state
            h0_dec = (torch.randn(self.num_layers, batch_size, self.hidden_size, dtype=torch.float, device=device),
                      torch.randn(self.num_layers, batch_size, self.hidden_size, dtype=torch.float, device=device))

            # Decoder produces sequence of distributions over output tokens
            # for each subsequence where at each step the current
            # conductor embedding is concatenated with the previous output
            # token to be used as input
            if use_teacher_forcing:
                embedding = embedding.expand(self.seq_length, batch_size, embedding.size(2))
                idx = range(embedding_idx * self.seq_length, embedding_idx * self.seq_length + self.seq_length)
                e = torch.cat((target[idx, :, :], embedding), dim=2).to(device)
                prev_note, h0_dec = self.lstm(e, h0_dec)
                prev_note = self.out(prev_note)
                out[idx, :, :] = prev_note
                prev_note = prev_note[-1, ::].unsqueeze(0)
            else:
                for note_idx in range(self.seq_length):
                    e = torch.cat((prev_note, embedding), -1)
                    prev_note, h0_dec = self.lstm(e, h0_dec)
                    prev_note = self.out(prev_note)

                    idx = embedding_idx * self.seq_length + note_idx
                    out[idx, :, :] = prev_note.squeeze()
        return out
    
    def reconstruct(self, latent, h0, c0, temperature):
        """
        Reconstruct the actual midi using categorical distribution
        """
        one_hot = torch.eye(self.input_size).to(device)
        batch_size = 1
        out = torch.zeros(self.max_seq_length, batch_size, self.input_size, dtype=torch.float, device=device)
        prev_note = torch.zeros(1, batch_size, self.input_size, dtype=torch.float, device=device)
        for embedding_idx in range(self.num_embeddings):
            embedding, (h0, c0) = self.conductor(latent.unsqueeze(0), (h0, c0))
            embedding = self.conductor_embeddings(embedding)
            h0_dec = (torch.randn(self.num_layers, batch_size, self.hidden_size, dtype=torch.float, device=device),
                      torch.randn(self.num_layers, batch_size, self.hidden_size, dtype=torch.float, device=device))
            for note_idx in range(self.seq_length):
                e = torch.cat((prev_note, embedding), -1)
                prev_note, h0_dec = self.lstm(e, h0_dec)
                prev_note = self.out(prev_note)
                prev_note = Categorical(prev_note / temperature).sample()
                prev_note = self.one_hot(prev_note)
                out[idx, :, :] = prev_note.squeeze()
        return out
                

    def init_hidden(self, batch_size=1):
        return (torch.zeros(self.num_layers, batch_size, self.hidden_size, dtype=torch.float, device=device),
                torch.zeros(self.num_layers, batch_size, self.hidden_size, dtype=torch.float, device=device))


class BiGRUEncoder(nn.Module):
    """
    Bi-directional GRU encoder from MusicVAE
    Inputs:
    - input_size:
    - hidden_size: hidden size of bidirectional gru
    - num_layers: Number of layers for bidirectional gru
    """

    def __init__(self,
                 input_size=61,
                 hidden_size=2048,
                 latent_size=512,
                 num_layers=2):
        super(BiGRUEncoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.num_layers = num_layers

        self.bigru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, bidirectional=True)
        self.mu = nn.Linear(in_features=2 * hidden_size, out_features=latent_size)
        self.sigma = nn.Linear(in_features=2 * hidden_size, out_features=latent_size)
        self.softplus = nn.Softplus()

    def forward(self, input, h0):
        batch_size = input.size(1)
        _, h_n = self.bigru(input, h0)
        h_n = h_n.view(self.num_layers, 2, batch_size, -1)[-1].view(batch_size, -1)
        mu = self.mu(h_n)
        sigma = self.softplus(self.sigma(h_n))
        return mu, sigma

    def init_hidden(self, batch_size=1):
        # Bidirectional gru so num_layers*2
        return torch.zeros(self.num_layers * 2, batch_size, self.hidden_size, dtype=torch.float, device=device)


class HierarchicalGRUDecoder(nn.Module):
    """
    Hierarchical decoder from MusicVAE
    """

    def __init__(self,
                 num_embeddings,
                 input_size=61,
                 hidden_size=1024,
                 latent_size=512,
                 num_layers=2,
                 max_seq_length=256,
                 seq_length=16):
        super(HierarchicalGRUDecoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.num_embeddings = num_embeddings
        self.max_seq_length = max_seq_length
        self.seq_length = seq_length
        self.num_layers = num_layers

        self.tanh = nn.Tanh()
        self.conductor = nn.GRU(input_size=latent_size, hidden_size=hidden_size, num_layers=num_layers)
        self.conductor_embeddings = nn.Sequential(
            nn.Linear(in_features=hidden_size, out_features=latent_size),
            nn.Tanh())
        self.gru = nn.GRU(input_size=input_size + latent_size, hidden_size=hidden_size, num_layers=num_layers)
        self.out = nn.Sequential(
            nn.Linear(in_features=hidden_size, out_features=input_size),
            nn.Softmax(dim=2)
        )

    def forward(self, target, latent, h0, use_teacher_forcing=True, temperature=1.0):
        batch_size = target.size(1)

        out = torch.zeros(self.max_seq_length, batch_size, self.input_size, dtype=torch.float, device=device)
        # Initialie start note
        prev_note = torch.zeros(1, batch_size, self.input_size, dtype=torch.float, device=device)

        # Conductor produces an embedding vector for each subsequence
        for embedding_idx in range(self.num_embeddings):
            embedding, h0 = self.conductor(latent.unsqueeze(0), h0)
            embedding = self.conductor_embeddings(embedding)

            # Initialize lower decoder hidden state
            h0_dec = torch.randn(self.num_layers, batch_size, self.hidden_size, dtype=torch.float, device=device)

            # Decoder produces sequence of distributions over output tokens
            # for each subsequence where at each step the current
            # conductor embedding is concatenated with the previous output
            # token to be used as input
            if use_teacher_forcing:
                embedding = embedding.expand(self.seq_length, batch_size, embedding.size(2)).to(device)
                idx = range(embedding_idx * self.seq_length, embedding_idx * self.seq_length + self.seq_length)
                e = torch.cat((target[idx, :, :], embedding), dim=2)
                prev_note, h0_dec = self.gru(e, h0_dec)
                prev_note = self.out(prev_note)
                out[idx, :, :] = prev_note
                prev_note = prev_note[-1, :, :].unsqueeze(0)
            else:
                for note_idx in range(self.seq_length):
                    e = torch.cat((prev_note, embedding), -1)
                    prev_note, h0_dec = self.gru(e, h0_dec)
                    prev_note = self.out(prev_note)

                    idx = embedding_idx * self.seq_length + note_idx
                    out[idx, :, :] = prev_note.squeeze()
        return out
    
    def reconstruct(self, latent, h0, temperature):
        """
        Reconstruct the actual midi using categorical distribution
        """
        one_hot = torch.eye(self.input_size).to(device)
        batch_size = h0.size(1)
        out = torch.zeros(self.max_seq_length, batch_size, self.input_size, dtype=torch.float, device=device)
        prev_note = torch.zeros(1, batch_size, self.input_size, dtype=torch.float, device=device)
        for embedding_idx in range(self.num_embeddings):
            embedding, h0 = self.conductor(latent.unsqueeze(0), h0)
            embedding = self.conductor_embeddings(embedding)
            h0_dec = torch.randn(self.num_layers, batch_size, self.hidden_size, dtype=torch.float, device=device)
            for note_idx in range(self.seq_length):
                e = torch.cat((prev_note, embedding), -1)
                prev_note, h0_dec = self.gru(e, h0_dec)
                prev_note = self.out(prev_note)
                prev_note = Categorical(prev_note / temperature).sample()
                prev_note = one_hot[prev_note]
                out[note_idx, :, :] = prev_note.squeeze()
        return out

    def init_hidden(self, batch_size=1):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size, dtype=torch.float, device=device)

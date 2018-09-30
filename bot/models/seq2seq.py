from torch import nn
import torch
import torch.nn.functional as F


class EncoderRNN(nn.Module):
    """
    Initializes a simple encoder consisting of an embedding layer and a GRU
    Expected to take in 1 word at a time
    """
    def __init__(self, input_size, hidden_size):
        """

        Parameters
        ----------
        input_size : int
        hidden_size : int
        """
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, data_in, hidden):
        """

        Parameters
        ----------
        data_in : pytorch tensor
            The data you wish to embed. Expects a single word.
        hidden : pytorch tensor
            The hidden state from the last call. If first call, send result of init_hidden

        Returns
        -------
        output : pytorch tensor
            The output of the gru
        hidden: pytorch tensor
            The hidden state from the GRU

        """
        embedded = self.embedding(data_in).view(1, 1, -1)
        return self.gru(embedded, hidden)

    def init_hidden(self, device):
        """

        Parameters
        ----------
        device : str
            Where cuda or CPU

        Returns
        -------
        pytorch tensor of zeros of correct size for GRU hidden state

        """
        return torch.zeros(1, 1, self.hidden_size).to(device)


class DecoderRNN(nn.Module):
    """
    A simple decoder which embeds a word, applies a relu, then returns the softmax output as well
    as the hidden state
    """
    def __init__(self, hidden_size, output_size):
        """

        Parameters
        ----------
        hidden_size : int
        output_size : int
        """
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, data_in, hidden):
        """

        Parameters
        ----------
        data_in : pytorch tensor
            The input to embed. Expects a single word
        hidden : pytorch tensor
            The hidden output from the GRU. For first round bast last hidden state from encoder.

        Returns
        -------
        output: pytorch tensor
            The output of the GRU passed through a softmax
        hidden: pytorch tensor
            The hidden state from the GRU

        """
        output = self.embedding(data_in).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def init_hidden(self, device):
        """

        Parameters
        ----------
        device : str
            Where cuda or CPU

        Returns
        -------
        pytorch tensor of zeros of correct size for GRU hidden state

        """
        return torch.zeros(1, 1, self.hidden_size).to(device)


class Seq2SeqModel:

    def __init__(self, input_size, output_size, hidden_size, lr=1e-3):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.encoder = EncoderRNN(input_size, hidden_size).to(self.device)
        self.decoder = DecoderRNN(hidden_size, output_size).to(self.device)

        self.encoder_optimizer = torch.optim.Adam(self.encoder.parameters(), lr)
        self.decoder_optimizer = torch.optim.Adam(self.decoder.parameters(), lr)

    def fit(self, pytorch_dataset, n_iters,):
        for iteration in range(1, n_iters+1):
            for query, answer in pytorch_dataset:
                loss = self.__train(query, answer)

    def __train(self, query, answer):
        encoder_hidden = self.encoder.init_hidden(self.device)

        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()

        input_length = query.size()[0]
        target_length = answer.size()[0]

        for i in range(input_length):
            encoder_output, encoder_hidden = self.encoder(query[i], encoder_hidden)

        decoder_hidden = encoder_hidden
        decoder_input


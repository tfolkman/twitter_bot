from torch.utils.data import Dataset
import pickle
import os


class TwitterDataset(Dataset):

    def __init__(self, data_directory):
        """

        Parameters
        ----------
        data_directory : str
            The directory which contains the pickled files:
            index2word, word2index, querylines, and answerlines
        """
        self.index2word = pickle.load(open(os.path.join(data_directory, "index2word.pickle"), "rb"))
        self.word2index = pickle.load( open(os.path.join(data_directory, "word2index.pickle"), "rb"))
        self.index_q = pickle.load(open(os.path.join(data_directory, "querylines.pickle"), "rb"))
        self.index_a = pickle.load(open(os.path.join(data_directory, "answerlines.pickle"), "rb"))

    def __len__(self):
        return len(self.index_q)

    def __getitem__(self, item):
        return self.index_q[item], self.index_a[item]

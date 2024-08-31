from torch.utils.data import Dataset
import torch

class DeepSepDataset(Dataset):
    def __init__(self, df, max_length, tokenizer):
        super().__init__()
        self.sequence = df['sequence']
        self.header = df['header']
        self.orf = df['ori_ORF']
        self.max_length = max_length
        self.tokenizer = tokenizer

    def __getitem__(self, index):
        sequence = self.sequence[index]
        header = self.header[index]
        orf = self.orf[index]
        tokenized_sequences = self.tokenizer.encode(sequence, max_length=self.max_length, truncation=True, padding='max_length')

        return torch.tensor(tokenized_sequences), header, orf

    def __len__(self):
        return len(self.sequence)
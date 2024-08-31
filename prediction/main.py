import torch
import os
import io
import torch.nn as nn
from Bio import SeqIO
from transformers import AutoTokenizer, AutoModel
from creat_ORF import *
from data_loader import DeepSepDataset
from torch.utils.data import DataLoader
from utils import run_DeepSep_neural_net
from homology import *
import argparse

def seq2kmer(seq):
    kmer = [seq[x:x + 3] for x in range(len(seq) + 1 - 3)]
    kmers = " ".join(kmer)
    return kmers

def filtered_sequence_func(seq):
    sequence_string = re.sub(r'\d+', '', seq)
    sequence_string = sequence_string.upper()
    sequence_string = re.sub(r'[^ATGCU]', '', sequence_string)
    sequence_string = sequence_string.replace('U', 'T')
    sequence_string = sequence_string.replace(' ', '')
    sequence_string = sequence_string.replace('\r\n', '\n')

    return sequence_string

class DeepSep_neural_net(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.bert = model
        self.cls = nn.Linear(768, 2)

    def forward(self, batch_seqs):
        output1, output2 = self.bert.forward(batch_seqs, return_dict=False, attention_mask=batch_seqs > 0)
        pre = self.cls(output2)

        return pre

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-sequence', help='str')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = AutoTokenizer.from_pretrained('../model/tokenizer')
    checkpoint = torch.load('../model/best_checkpoint.pth.tar', map_location=torch.device(device))
    model = DeepSep_neural_net(AutoModel.from_pretrained('../model/checkpoint-11007')).to(device)
    model.load_state_dict(checkpoint['state_dict'])

    sequence = str(args.sequence)
    sequence_string = filtered_sequence_func(sequence)

    if not sequence_string.startswith('>'):
        sequence_string = '>example\n' + sequence_string

    sequence_file = io.StringIO(sequence_string)
    fasta_sequence = SeqIO.read(sequence_file, "fasta")

    sequence = str(fasta_sequence.seq)
    id = '>' + str(fasta_sequence.id)

    prefix_location = f'./tmp'

    df_nt_sentence, df_nr_sentence = creat_ORF_main_func(sequence, id, f'{prefix_location}/ORFs')
    df_nt_sentence['sequence'] = df_nt_sentence['sequence'].apply(lambda x: seq2kmer(str(x)))

    DeepSep_dataset = DeepSepDataset(df_nt_sentence, max_length=300, tokenizer=tokenizer)
    DeepSep_dataloader = DataLoader(DeepSep_dataset, batch_size=1024, shuffle=False)

    dl_result = run_DeepSep_neural_net(model, DeepSep_dataloader, device)
    dl_result.to_csv(os.path.join(prefix_location, 'dl_result.csv'), index=False)

    diamond_input = f'{prefix_location}/homology/diamond_input.fasta'
    diamond_pred_output = f'{prefix_location}/homology/diamond_pred_result.txt'
    result_output_dir = f'{prefix_location}/homology'

    if not os.path.exists(result_output_dir):
        os.makedirs(result_output_dir)

    new_df_nr = get_queries(dl_result, df_nr_sentence, diamond_input)

    diamond_func(diamond_input, diamond_pred_output, threads=160)

    with open(diamond_pred_output, 'r') as file:
        if file.readline():
            preliminary_result = analysis(result_output_dir, diamond_pred_output)
        



import torch
import os
import io
import torch.nn as nn
from Bio import SeqIO
from transformers import AutoTokenizer, AutoModel, logging
logging.set_verbosity_error()
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
    parser.add_argument('-sequence_file', type=str, help='path to FASTA file')
    parser.add_argument('-diamond_program_path', help='path', required=True)
    args = parser.parse_args()

    prefix_location = f'./tmp'

    try:
        fasta_record = list(SeqIO.parse(args.sequence_file, "fasta"))
    except FileNotFoundError:
        parser.error(f"File not found: {args.sequence_file}")
    except ValueError:
        parser.error(f"Invalid FASTA format: {args.sequence_file}")

    df_sentences = creat_ORF_main_func(fasta_record, f'{prefix_location}/ORFs')

    df_sentences['sequence'] = df_sentences['sequence'].apply(lambda x: seq2kmer(str(x)))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    tokenizer = AutoTokenizer.from_pretrained('../model/tokenizer')
    model = DeepSep_neural_net(AutoModel.from_pretrained('../model/checkpoint-11007')).to(device)
    checkpoint = torch.load('../model/best_checkpoint.pth.tar', map_location=torch.device(device))
    model.load_state_dict(checkpoint['state_dict'])

    DeepSep_dataset = DeepSepDataset(df_sentences, max_length=300, tokenizer=tokenizer)
    DeepSep_dataloader = DataLoader(DeepSep_dataset, batch_size=1024, shuffle=False)

    dl_result = run_DeepSep_neural_net(model, DeepSep_dataloader, device)
    dl_result.to_csv(os.path.join(prefix_location, 'dl_result.csv'), index=False)

    diamond_input = f'{prefix_location}/homology/diamond_input.fasta'
    diamond_pred_output = f'{prefix_location}/homology/diamond_pred_result.txt'
    result_output_dir = f'{prefix_location}/homology'

    if not os.path.exists(result_output_dir):
        os.makedirs(result_output_dir)

    new_df_nr = get_queries(dl_result, df_sentences, diamond_input)

    with open(diamond_input, 'r') as diamond_input_file:
        if diamond_input_file.readline():
            diamond_func(diamond_input, diamond_pred_output, args.diamond_program_path, threads=160)

            with open(diamond_pred_output, 'r') as file:
                if file.readline():
                    preliminary_result = analysis(result_output_dir, diamond_pred_output)

                    DeepSep_result = df_sentences[df_sentences['header'].isin(preliminary_result['qseqid'])]

                    DeepSep_result[['header', 'nt', 'nr']].to_csv(f'{prefix_location}/final_results.csv', index=False)

                else:
                    with open(f'{prefix_location}/final_results.txt', 'w') as final_file:
                        final_file.write('Not Find Any Selenoproteins.')

        else:
            raise ValueError("DL model did not predict any possible candidates.")

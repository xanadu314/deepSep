import pandas as pd
import os
import numpy as np
import torch.cuda
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer
import torch.nn as nn
from tqdm import tqdm
import random
import time
from metrics import cofusion_matrix, sensitivity, specificity, auc, mcc, accuracy, precision, recall, f1, AUPRC
import argparse

def seq2kmer(seq, k):
    kmer = [seq[x:x + k] for x in range(len(seq) + 1 - k)]
    kmers = " ".join(kmer)
    return kmers

class MyDataset(Dataset):
    def __init__(self, df):
        super().__init__()
        self.sequence = df['sequence']
        self.label = df['label']
        self.header = df['header']
        self.orf = df['ori_ORF']

    def __getitem__(self, index):
        sequence = self.sequence[index]
        label = self.label[index]
        header = self.header[index]
        orf = self.orf[index]
        tokenized_sequences = tokenizer.encode(sequence, max_length=max_length, truncation=True, padding='max_length')

        return torch.tensor(tokenized_sequences), label, header, orf

    def __len__(self):
        return len(self.label)

class Mymodel(torch.nn.Module):
    def __init__(self, model, in_features):
        super().__init__()
        self.bert = model
        self.cls = nn.Linear(in_features, 2)
        # self.loss = nn.CrossEntropyLoss()

    def forward(self, batch_seqs, batch_labels=None):
        output1, output2 = self.bert.forward(batch_seqs, return_dict=False, attention_mask=batch_seqs > 0)
        pre = self.cls(output2)

        return pre

if __name__ == '__main__':
    start_time = time.time()

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    torch.backends.cudnn.deterministic = True

    parser = argparse.ArgumentParser()
    parser.add_argument('-k', help='num')
    parser.add_argument('-test_df', help='path')
    parser.add_argument('-tokenizer', help='path')
    parser.add_argument('-finetuned_model', help='path')
    parser.add_argument('-pretrained_model', help='path')
    parser.add_argument('-output_path', help='path')
    parser.add_argument('-max_length', help='num')
    parser.add_argument('-hidden_size', help='num')
    args = parser.parse_args()

    hidden_size = int(args.hidden_size)

    test_raw_df = pd.read_csv(args.test_df)

    test_df = test_raw_df.loc[:, :]
    test_df['sequence'] = test_df['sequence'].apply(lambda x: seq2kmer(str(x), int(args.k)))

    max_length = int(args.max_length)

    test_dataset = MyDataset(test_df)
    test_dataloader = DataLoader(test_dataset, batch_size=1024, shuffle=False)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model_dir = args.finetuned_model
    checkpoint = torch.load(model_dir + '/' + 'best_checkpoint.pth.tar')
    print('---------load_model: best_checkpoint.pth.tar----->>>>')
    model = Mymodel(AutoModel.from_pretrained(args.pretrained_model), hidden_size).to(device) # load pre-trained model

    print('model loaded ---->>')
    model.load_state_dict(checkpoint['state_dict'])

    loss = nn.CrossEntropyLoss()
    # opt = torch.optim.AdamW(model.parameters(), lr=lr) # , weight_decay=0.0025
    metrics_dict = {"sensitivity": sensitivity, "specificity": specificity, "accuracy": accuracy, "MCC": mcc,
                    "AUC": auc, "precision": precision, "recall": recall, "f1": f1, "AUPRC": AUPRC}

    test_outputdir = args.output_path
    if os.path.exists(test_outputdir):
        print('TEST Outputdir is exitsted')
    else:
        os.makedirs(test_outputdir)
        print('success to create dir')

    test_result = pd.DataFrame()
    all_test_acc = []
    with open(os.path.join(test_outputdir, 'test_performance.txt'), 'w') as f:
        print(test_outputdir, file=f, flush=True)
        print("The number of testing data:" + str(len(test_dataset)), file=f, flush=True)

        all_test_loss = []
        test_pre_values, test_preds, test_labels, test_headers, test_orf_seqs = [], [], [], [], []
        right_num = 0
        model.eval()
        with torch.no_grad():
            for batch_sequences, batch_labels, batch_headers, batch_orf_seqs in tqdm(test_dataloader):
                test_pre = model.forward(batch_sequences.to(device))
                test_loss = loss(test_pre, batch_labels.to(device)).to(device)
                all_test_loss.append(test_loss.item())
                test_pred = torch.argmax(test_pre, dim=-1)
                right_num += torch.sum(test_pred == batch_labels.to(device)).item()
                probs = torch.softmax(test_pre, dim=-1)
                
                # 选择正类的概率值
                scores = probs[:, 1]
                test_pre_values.append(scores.cpu()), test_preds.append(test_pred.cpu()), test_labels.append(batch_labels), test_headers.append(batch_headers), test_orf_seqs.append(batch_orf_seqs)
        ave_test_loss = sum(all_test_loss) / len(test_dataloader)
        acc = right_num / len(test_dataset)
        all_test_acc.append(acc)
        test_pre_values, test_preds, test_labels = torch.cat(test_pre_values), torch.cat(test_preds), torch.cat(test_labels)

        print(f'test_loss: {ave_test_loss:.3f}, test_acc: {right_num}/{len(test_dataset)} {acc:.3f}', file=f, flush=True)
        print(f'test_loss: {ave_test_loss:.3f}, test_acc: {acc:.3f}')
        for key in metrics_dict.keys():
            if (key != "AUC" and key != "AUPRC"):
                metrics = metrics_dict[key](test_labels, test_preds)
            else:
                metrics = metrics_dict[key](test_labels, test_pre_values)
            print(f"test_{key}: {metrics:.3f}", file=f, flush=True)

        tn_t, fp_t, fn_t, tp_t = cofusion_matrix(test_labels, test_preds)
        print("test_true_negative: value: %d" % (tn_t), file=f, flush=True)
        print("test_false_positive: value: %d" % (fp_t), file=f, flush=True)
        print("test_false_negative: value: %d" % (fn_t), file=f, flush=True)
        print("test_true_positive: value: %d" % (tp_t), file=f, flush=True)

    test_headers_values = [item for tup in test_headers for item in tup]
    test_test_orf_seqs_values = [item for tup in test_orf_seqs for item in tup]
    test_result['header'], test_result['orf_sequence'], test_result['pre_value'], test_result['pre_label'], test_result['truth_label'] = test_headers_values, test_test_orf_seqs_values, test_pre_values.numpy().tolist(), test_preds, test_labels

    test_result.to_csv(os.path.join(test_outputdir, 'test_result.csv'), index=False)

    end_time = time.time()
    total_time = end_time - start_time

    print(f'共耗时：{total_time:.3f} s, or {(total_time / 3600):.2f} hours')
    print("Done!")
    # analyze(temp=)

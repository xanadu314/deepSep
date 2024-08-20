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
from torch.utils.tensorboard import SummaryWriter
import argparse

def seq2kmer(seq, k):
    kmer = [seq[x:x + k] for x in range(len(seq) + 1 - k)]
    kmers = " ".join(kmer)
    return kmers

def save_checkpoint(state, is_best, OutputDir):
    if is_best:
        print('=> Saving a new best from epoch %d' % state['epoch'])
        torch.save(state, OutputDir + '/' + 'best_checkpoint.pth.tar')
    else:
        print("=> Validation Performance did not improve")


class MyDataset(Dataset):
    def __init__(self, df):
        super().__init__()
        self.sequence = df['sequence']
        self.label = df['label']

    def __getitem__(self, index):
        sequence = self.sequence[index]
        label = self.label[index]
        tokenized_sequences = tokenizer.encode(sequence, max_length=max_length, truncation=True, padding='max_length')

        return torch.tensor(tokenized_sequences), label

    def __len__(self):
        return len(self.label)

class Mymodel(torch.nn.Module):
    def __init__(self, model, in_features):
        super().__init__()
        self.bert = model
        self.cls = nn.Linear(in_features, 2)

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
    parser.add_argument('-train_data', help='df')
    parser.add_argument('-valid_data', help='df')
    parser.add_argument('-tokenizer_path', help='path')
    parser.add_argument('-model', help='path')
    parser.add_argument('-save_path', help='path')
    parser.add_argument('-k', help='num')
    parser.add_argument('-lr', help='num')
    parser.add_argument('-max_length', help='num')
    parser.add_argument('-hidden_size', help='num')
    args = parser.parse_args()

    train_data = args.train_data
    valid_data = args.valid_data
    tokenizer_path = args.tokenizer_path
    model = args.model
    save_path = args.save_path
    lr = float(args.lr)
    k = int(args.k)
    hidden_size = int(args.hidden_size)

    train_df = pd.read_csv(train_data)
    eval_df = pd.read_csv(valid_data)

    max_length = int(args.max_length)

    train_df['sequence'] = train_df['sequence'].apply(lambda x: seq2kmer(str(x), k))
    eval_df['sequence'] = eval_df['sequence'].apply(lambda x: seq2kmer(str(x), k))

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    train_dataset = MyDataset(train_df)
    train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)

    eval_dataset = MyDataset(eval_df)
    eval_dataloader = DataLoader(eval_dataset, batch_size=256, shuffle=True)

    epoch = 10
    lr = lr
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = Mymodel(AutoModel.from_pretrained(model), hidden_size).to(device)
    loss = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=lr) # , weight_decay=0.0025
    metrics_dict = {"sensitivity": sensitivity, "specificity": specificity, "accuracy": accuracy, "MCC": mcc,
                    "AUC": auc, "precision": precision, "recall": recall, "f1": f1, "AUPRC": AUPRC}

    outputdir = save_path  # os.path.join(path, model_name, 'train_valid_result')
    if os.path.exists(outputdir):
        print('Outputdir is exitsted')
    else:
        os.makedirs(outputdir)
        print('success to create dir')

    tensorboard_writer = SummaryWriter(log_dir=outputdir + '/' + 'model' + '_' + time.strftime('%m-%d_%H.%M', time.localtime()) + '_' + str(epoch))
    all_epochs_train_losses = []
    all_epochs_val_losses = []
    all_val_acc = []
    best_acc = 0
    with open(os.path.join(outputdir, 'train_valid_performance.txt'), 'w') as f:
        print(outputdir, 'lr:' + str(lr), 'epoch:' + str(int(epoch)), file=f, flush=True)
        print("The number of training data:" + str(len(train_dataset)), file=f, flush=True)
        print("The number of validation data:" + str(len(eval_dataset)), file=f, flush=True)

        for i in range(epoch):
            train_losses = []
            train_pre_values, train_preds, train_labels = [], [], []
            model.train()
            for batch_sequences, batch_labels in tqdm(train_dataloader):
                train_pre = model.forward(batch_sequences.to(device), batch_labels.to(device))
                train_loss = loss(train_pre, batch_labels.to(device)).to(device)
                train_losses.append(train_loss.item())
                train_pred = torch.argmax(train_pre, dim=-1) # 索引
                train_probs = torch.softmax(train_pre, dim=-1) # value
                # 选择正类的概率值
                train_scores = train_probs[:, 1] # value
                train_pre_values.append(train_scores.cpu()), train_preds.append(train_pred.cpu()), train_labels.append(batch_labels)
                train_loss.backward()
                opt.step()
                opt.zero_grad()

            per_epoch_train_loss = sum(train_losses) / len(train_dataloader)
            all_epochs_train_losses.append(per_epoch_train_loss)
            train_pre_values, train_preds, train_labels = torch.cat(train_pre_values), torch.cat(train_preds), torch.cat(train_labels)
            train_pre_values = train_pre_values.detach().numpy()

            print(f'\ntrain_epoch: {i+1}, epoch_loss: {per_epoch_train_loss:.3f}', file=f, flush=True)
            print(f'train_epoch: {i+1}, epoch_loss: {per_epoch_train_loss:.3f}')
            for key in metrics_dict.keys():
                if (key != "AUC" and key != "AUPRC"):
                    metrics = metrics_dict[key](train_labels, train_preds)
                else:
                    metrics = metrics_dict[key](train_labels, train_pre_values)
                print(f"train_{key}: {metrics:.3f}", file=f, flush=True)
            tn_t, fp_t, fn_t, tp_t = cofusion_matrix(train_labels, train_preds)
            print("train_true_negative: value: %d, epoch: %d" % (tn_t, i + 1), file=f, flush=True)
            print("train_false_positive: value: %d, epoch: %d" % (fp_t, i + 1), file=f, flush=True)
            print("train_false_negative: value: %d, epoch: %d" % (fn_t, i + 1), file=f, flush=True)
            print("train_true_positive: value: %d, epoch: %d" % (tp_t, i + 1), file=f, flush=True)

            print("validation...", file=f, flush=True)

            all_eval_loss = []
            eval_pre_values, eval_preds, eval_labels = [], [], []
            right_num = 0
            model.eval()
            with torch.no_grad():
                for batch_sequences, batch_labels in tqdm(eval_dataloader):
                    eval_pre = model.forward(batch_sequences.to(device))
                    eval_loss = loss(eval_pre, batch_labels.to(device)).to(device)
                    all_eval_loss.append(eval_loss.item())
                    eval_pred = torch.argmax(eval_pre, dim=-1)
                    eval_probs = torch.softmax(eval_pre, dim=-1)  # value
                    # 选择正类的概率值
                    eval_scores = eval_probs[:, 1] # value
                    right_num += torch.sum(eval_pred == batch_labels.to(device)).item()
                    eval_pre_values.append(eval_scores.cpu()), eval_preds.append(eval_pred.cpu()), eval_labels.append(batch_labels)

            per_epoch_eval_loss = sum(all_eval_loss) / len(eval_dataloader)
            all_epochs_val_losses.append(per_epoch_eval_loss)
            per_epoch_eval_acc = right_num / len(eval_dataset)
            all_val_acc.append(per_epoch_eval_acc)
            eval_pre_values, eval_preds, eval_labels = torch.cat(eval_pre_values), torch.cat(eval_preds), torch.cat(eval_labels)
            eval_pre_values = eval_pre_values.detach().numpy()

            print(f'eval_epoch: {i+1}, epoch_loss: {per_epoch_eval_loss:.3f}, epoch_acc: {right_num}/{len(eval_dataset)} {per_epoch_eval_acc:.3f}', file=f, flush=True)
            print(f'eval_epoch: {i+1}, epoch_acc: {per_epoch_eval_acc:.3f}')
            for key in metrics_dict.keys():
                if (key != "AUC" and key != "AUPRC"):
                    metrics = metrics_dict[key](eval_labels, eval_preds)
                else:
                    metrics = metrics_dict[key](eval_labels, eval_pre_values)
                print(f"valid_{key}: {metrics:.3f}", file=f, flush=True)

            tn_t, fp_t, fn_t, tp_t = cofusion_matrix(eval_labels, eval_preds)
            print("validation_true_negative: value: %d, epoch: %d" % (tn_t, i + 1), file=f, flush=True)
            print("validation_false_positive: value: %d, epoch: %d" % (fp_t, i + 1), file=f, flush=True)
            print("validation_false_negative: value: %d, epoch: %d" % (fn_t, i + 1), file=f, flush=True)
            print("validation_true_positive: value: %d, epoch: %d" % (tp_t, i + 1), file=f, flush=True)

            tensorboard_writer.add_scalar('loss/train', per_epoch_train_loss, i+1)
            tensorboard_writer.add_scalar('loss/eval', per_epoch_eval_loss, i+1)
            tensorboard_writer.add_scalar('acc/eval', per_epoch_eval_acc, i+1)

            cur_acc = per_epoch_eval_acc
            is_best = bool(cur_acc >= best_acc)
            best_acc = max(cur_acc, best_acc)

            if is_best:
                torch_val_value_best = eval_pre_values
                torch_val_best = eval_preds
                torch_val_y_best = eval_labels
                best_epoch = i+1

            save_checkpoint({
                'epoch': i + 1,
                'state_dict': model.state_dict(),
                'best_accuracy': best_acc,
                'optimizer': opt.state_dict()
            }, is_best, outputdir)

        ave_train_loss = sum(all_epochs_val_losses)/epoch
        ave_val_loss = sum(all_epochs_val_losses)/epoch
        ave_val_acc = sum(all_val_acc)/epoch
        print(f'total_epoch: {epoch}, ave_train_loss: {ave_train_loss:.3f}, ave_val_loss: {ave_val_loss:.3f}, ave_val_acc: {ave_val_acc:.3f}', file=f, flush=True)

        print("\nBest Validation...\n", file=f, flush=True)

        print(f"Best Validation is epoch: {best_epoch}", file=f, flush=True)
        for key in metrics_dict.keys():
            if (key != "AUC" and key != "AUPRC"):
                best_val_metrics = metrics_dict[key](torch_val_y_best, torch_val_best)
            else:
                best_val_metrics = metrics_dict[key](torch_val_y_best, torch_val_value_best)
            print(f"best_validation_{key}: {best_val_metrics:.3f}", file=f, flush=True)
        print(f"Best validation is epoch: {best_epoch}")

    tensorboard_writer.close()
    end_time = time.time()
    total_time = end_time - start_time

    print(f'共耗时：{total_time:.3f} s, or {(total_time / 3600):.2f} hours')
    print("Done!")

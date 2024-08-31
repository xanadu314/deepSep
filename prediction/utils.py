import torch
from tqdm import tqdm
import pandas as pd
import logging
import os

def run_DeepSep_neural_net(model, DeepSep_dataloader, device):
    test_pre_values, test_pre_labels, test_headers, test_orf_seqs = [], [], [], []
    test_result = pd.DataFrame()

    model.eval() # training session with train dataset
    with torch.no_grad():
        logging.info('Deep leanrning prediction starts on the dataset')
        for batch_sequences, batch_headers, batch_orf_seqs in tqdm(DeepSep_dataloader):
            output = model.forward(batch_sequences.to(device))
            test_pred_label = torch.argmax(output, dim=-1)
            probs = torch.softmax(output, dim=-1)
            probabilities = probs[:, 1]

            test_pre_values.append(probabilities.cpu()), test_pre_labels.append(test_pred_label.cpu()),
            test_headers.append(batch_headers), test_orf_seqs.append(batch_orf_seqs)

        test_pre_values, test_pre_labels = torch.cat(test_pre_values), torch.cat(test_pre_labels)

        test_headers_values = [item for tup in test_headers for item in tup]
        test_orf_seqs_values = [item for tup in test_orf_seqs for item in tup]
        test_result['header'], test_result['orf_sequence'], test_result['pre_value'], test_result['pre_label'] = test_headers_values, test_orf_seqs_values, test_pre_values.numpy().tolist(), test_pre_labels

        logging.info('Deep learning prediction ended on test dataset')

    return test_result

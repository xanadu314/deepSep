import os
from Bio import SeqIO as seqio
from Bio.Seq import Seq
import warnings
import pandas as pd

def find_STAR_CODON(left):
    p1 = left[::-1]

    ##先判断终止密码子的位置
    for i in range(0, len(p1), 3):
        if i + 2 < len(p1):
            codon = p1[i] + p1[i + 1] + p1[i + 2]
            if codon in ['GAT', 'AAT', 'AGT']:  # (反)终止密码子
                left_str = p1[:i]
                break

    if 'left_str' in dir():  # 先确定终止密码子的范围
        pass
    else:
        left_str = p1  # 如果都没碰到终止密码子，那就直接定为p1

    # 在确定终止密码子的范围内找起始密码子
    initial_codon_tank = []
    for i in range(0, len(left_str), 3):
        if i + 2 < len(left_str):
            codon = left_str[i] + left_str[i + 1] + left_str[i + 2]
            if codon in ['GTA', 'GTG', 'GTT']:
                initial_codon_tank.append(i + 3)

    if len(initial_codon_tank) == 0:  # 如果没有起始密码子，这条序列就不要
        return None
    else:  # 如果有起始密码子
        initial_codon_l = initial_codon_tank[-1]  # 取最远的那个起始密码子：initial_codon_tank[-1]
        up_stream = left_str[:initial_codon_l][::-1]

        return up_stream

def find_STOP_CODON(right_str):
    for i in range(0, len(right_str), 3):
        if i + 2 < len(right_str):
            codon = right_str[i] + right_str[i + 1] + right_str[i + 2]
            if codon in ['TAG', 'TAA', 'TGA']:
                down_stream = right_str[:i+3]
                return down_stream

    if 'down_stream' in dir():
        pass
    else:
        down_stream = right_str

    return down_stream

def ORF_Frame(seq, id, sign):
    result_sequences = []

    for frame in [1, 2, 3]:
        if frame == 1:
            start_i = 0
        elif frame == 2:
            start_i = 1
        else:
            start_i = 2

        for i in range(start_i, len(seq), 3):
            if i + 2 < len(seq):
                codon = seq[i] + seq[i + 1] + seq[i + 2]
                if codon == 'TGA':
                    left_str = seq[:i] # 上游
                    right_str = seq[i+3:] # 下游

                    up_stream = find_STAR_CODON(left_str)
                    if up_stream is None:
                        continue

                    down_stream = find_STOP_CODON(right_str)

                    orf_start = i - len(up_stream)
                    orf_end = i + len(down_stream) + 3
                    orf = up_stream + codon + down_stream

                    # 截取
                    t_seq = right_str[:300] # 下游 直接取TGA后个300个碱基

                    if orf == seq[orf_start: orf_end]:
                        aa_seq = translate(orf)
                        aa_seq = str(aa_seq)

                        result_sequences.append((id, orf_start, orf_end, sign, frame, orf, aa_seq, t_seq))

                    # else:
                    #     print('wrong seq')

    return result_sequences

def translate(seq_line):
    seq_line = seq_line.strip()
    seq_line = Seq(seq_line)
    warnings.filterwarnings("ignore")
    seq_line = seq_line.translate()
    if seq_line[-1] == '*':
        seq_line = str(seq_line)[::-1].replace('*', '', 1)[::-1].strip()
    seq_line = str(seq_line).replace('*', 'U')
    return seq_line

def write_to_file(all_results, save_path):
    csv_data = []
    nt_fasta_data = []
    nr_fasta_data = []

    for results in all_results:
        all_seqs = results
        for orf_id, orf_start, orf_end, strand_sign, frame, orf, aa_seq, t_seq in all_seqs:
            csv_data.append(
                f'>{orf_id}_{orf_start}_{orf_end}_Frame:{strand_sign}{frame}\t{orf}\t{aa_seq}\t{t_seq}\n')
            nt_fasta_data.append(f'>{orf_id}_{orf_start}_{orf_end}_Frame:{strand_sign}{frame}\n{orf}\n')
            nr_fasta_data.append(
                f'>{orf_id}_{orf_start}_{orf_end}_Frame:{strand_sign}{frame}\n{aa_seq}\n')

    data_rows = [line.split('\t') for line in csv_data]
    # 批量写入 ORFs.csv
    whole_result = pd.DataFrame(data_rows, columns=['header', 'nt', 'nr', 'sequence'])
    whole_result.to_csv(os.path.join(save_path, 'ORFs.csv'), index=False)

    # 批量写入 ORFs_NT.fasta
    with open(os.path.join(save_path, 'ORFs_NT.fasta'), 'w', encoding='utf-8') as f_nt_fasta:
        f_nt_fasta.writelines(nt_fasta_data)

    # 批量写入 ORFs_NR.fasta
    with open(os.path.join(save_path, 'ORFs_NR.fasta'), 'w', encoding='utf-8') as f_nr_fasta:
        f_nr_fasta.writelines(nr_fasta_data)

    return whole_result

def creat_ORF_main_func(records, save_path):
    output_path = save_path

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    results = []
    reversed_results = []
    for record in records:
        sequence = str(record.seq)
        id = str(record.description).replace(' ', '_')

        result = ORF_Frame(sequence, id, sign='+')
        results.append(result)

        reversed_sequence = Seq(sequence).reverse_complement()
        reversed_result = ORF_Frame(reversed_sequence, id, sign='-')
        reversed_results.append(reversed_result)

    whole_results = results+reversed_results

    results = write_to_file(whole_results, save_path)

    return results
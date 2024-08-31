import os
import pandas as pd
from Bio.Seq import Seq
import warnings

def find_STAR_CODON(left):
    p1 = left[::-1]

    for i in range(0, len(p1), 3):
        if i + 2 < len(p1):
            codon = p1[i] + p1[i + 1] + p1[i + 2]
            if codon in ['GAT', 'AAT', 'AGT']:
                left_str = p1[:i]
                break

    if 'left_str' in dir():
        pass
    else:
        left_str = p1

    initial_codon_tank = []
    for i in range(0, len(left_str), 3):
        if i + 2 < len(left_str):
            codon = left_str[i] + left_str[i + 1] + left_str[i + 2]
            if codon in ['GTA', 'GTG', 'GTT']:
                initial_codon_tank.append(i + 3)

    if len(initial_codon_tank) == 0:
        return None
    else:
        initial_codon_l = initial_codon_tank[-1]
        up_stream = left_str[:initial_codon_l][::-1]
        return up_stream

def find_STOP_CODON(right_str):
    for i in range(0, len(right_str), 3):
        if i + 2 < len(right_str):
            codon = right_str[i] + right_str[i + 1] + right_str[i + 2]
            if codon in ['TAG', 'TAA', 'TGA']:
                down_stream = right_str[:i+3]
                return down_stream

    return None

def ORF_Frame(seq, id, num, frame):
    nt_sequences = []
    nr_sequences = []

    tga_count = 0
    if num == 1:
        start_i = 0
    elif num == 2:
        start_i = 1
    else:
        start_i = 2

    for i in range(start_i, len(seq), 3):
        if i + 2 < len(seq):
            codon = seq[i] + seq[i + 1] + seq[i + 2]
            if codon == 'TGA':
                left_str = seq[:i]
                right_str = seq[i+3:]
                tga_count += 1

                up_stream = find_STAR_CODON(left_str)
                if up_stream is None:
                    continue

                down_stream = find_STOP_CODON(right_str)
                if down_stream is None:
                    continue

                orf_start = i - len(up_stream)
                orf_end = i + len(down_stream) + 3
                orf = up_stream + codon + down_stream

                # 截取
                t_seq = right_str[:300]

                if orf == seq[orf_start: orf_end]:
                    tga_loc = len(up_stream)
                    nt_sequences.append((id, orf_start, orf_end, orf, frame, t_seq, tga_loc))

                    aa_seq = translate(orf)
                    aa_seq = str(aa_seq)

                    u_loc = int(((tga_loc + 3) / 3) - 1)

                    nr_sequences.append((id, orf_start, orf_end, aa_seq, frame, '', u_loc))
                else:
                    print('wrong seq')

    print(f'frame: {frame}, tga_count: {tga_count}')
    return nt_sequences, nr_sequences, tga_count

def translate(seq_line):
    seq_line = seq_line.strip()
    seq_line = Seq(seq_line)
    warnings.filterwarnings("ignore")
    seq_line = seq_line.translate()
    if seq_line[-1] == '*':
        seq_line = str(seq_line)[::-1].replace('*', '', 1)[::-1].strip()
    return seq_line

def write_to_file(nt_sequences, nr_sequences, nt_fasta_filename, nr_fasta_filename, nt_csv_filename, nr_csv_filename):
    with open(nt_fasta_filename, 'w') as file:
        for tup in nt_sequences:
            for header, start, stop, sequence, frame, t_Seq, tga_loc in tup:
                file.write(f'{header}_{start}_{stop}_frame:{frame}\n{sequence}\n')
    with open(nr_fasta_filename, 'w') as file:
        for tup in nr_sequences:
            for header, start, stop, sequence, frame, t_Seq, u_loc in tup:
                file.write(f'{header}_{start}_{stop}_frame:{frame}\n{sequence}\n')

    # csv
    with open(nt_csv_filename, 'w') as file:
        for tup in nt_sequences:
            for header, start, stop, sequence, frame, t_Seq, tga_loc in tup:
                file.write(f'{header}_{start}_{stop}_frame:{frame}\t{tga_loc}\t{sequence}\t{t_Seq}\n')
    with open(nr_csv_filename, 'w') as file:
        for tup in nr_sequences:
            for header, start, stop, sequence, frame, t_Seq, u_loc in tup:
                file.write(f'{header}_{start}_{stop}_frame:{frame}\t{u_loc}\t{sequence}\n')

    nt_ORFs_df = pd.read_csv(nt_csv_filename, sep='\t', names=['header', 'tga_loc', 'ori_ORF', 'sequence'])
    nr_ORFs_df = pd.read_csv(nr_csv_filename, sep='\t', names=['header', 'u_loc', 'ori_ORF'])

    return nt_ORFs_df, nr_ORFs_df

def creat_ORF_main_func(sequence, id, save_path):
    output_path = save_path

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    all_forward_nt_seqs = []
    all_forward_nr_seqs = []
    all_forward_tgas = []

    all_reversed_nt_seqs = []
    all_reversed_nr_seqs = []
    all_reversed_tgas = []

    for n in [1, 2, 3]:
        nt_seqs, nr_seqs, tga = ORF_Frame(sequence, id, n, '+'+str(n))

        all_forward_nt_seqs.append(nt_seqs)
        all_forward_nr_seqs.append(nr_seqs)
        all_forward_tgas.append(tga)

        reversed_sequence = Seq(sequence).reverse_complement()
        reversed_nt_seqs, reversed_nr_seqs, reversed_tga = ORF_Frame(reversed_sequence, id, n, '-'+str(n))

        all_reversed_nt_seqs.append(reversed_nt_seqs)
        all_reversed_nr_seqs.append(reversed_nr_seqs)
        all_reversed_tgas.append(reversed_tga)

    forward_nt_df, forward_nr_df = write_to_file(all_forward_nt_seqs, all_forward_nr_seqs,
                  nt_fasta_filename=output_path + '/nt_frame.fasta',
                  nr_fasta_filename=output_path + '/nr_frame.fasta',
                  nt_csv_filename=output_path + '/nt_frame.csv',
                  nr_csv_filename=output_path + '/nr_frame.csv',)

    reversed_nt_df, reversed_nr_df = write_to_file(all_reversed_nt_seqs, all_reversed_nr_seqs,
                  nt_fasta_filename=output_path + '/reversed_nt_frame.fasta',
                  nr_fasta_filename=output_path + '/reversed_nr_frame.fasta',
                  nt_csv_filename=output_path + '/reversed_nt_frame.csv',
                  nr_csv_filename=output_path + '/reversed_nr_frame.csv',)


    all_nt_seqs = pd.concat([forward_nt_df, reversed_nt_df])
    all_nr_seqs = pd.concat([forward_nr_df, reversed_nr_df])

    all_nt_seqs = all_nt_seqs.reset_index(drop=True)
    all_nr_seqs = all_nr_seqs.reset_index(drop=True)

    return all_nt_seqs, all_nr_seqs

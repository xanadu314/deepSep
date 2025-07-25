import logging
import os
import pandas as pd
import re


def get_queries(diamond_input, whole_df, queries_file_name):
    pre_pos = diamond_input[diamond_input['pre_label'] == 1]

    pre_pos['header'] = pre_pos['header'].apply(lambda x: '>' + str(x) if '>' not in str(x) else str(x))
    whole_df['header'] = whole_df['header'].apply(lambda x: '>' + str(x) if '>' not in str(x) else str(x))

    pre_pos_nr = whole_df[whole_df['header'].isin(pre_pos['header'])]

    new_pre_pos_nr = pre_pos_nr.loc[:, :]

    new_pre_pos_nr['nr'] = new_pre_pos_nr['nr'].apply(lambda x: str(x).replace('*', 'U'))

    with open(queries_file_name, 'w') as f:
        for _, row in new_pre_pos_nr.iterrows():
            f.write(row['header'] + '\n' + row['nr'] + '\n')

    return new_pre_pos_nr

def diamond_func(query, out, program_path, threads):
    logging.info('DIAMOND prediction starts on the dataset')

    program = program_path # './diamond_v2.1.8/diamond'
    # 修改真实数据库路径
    db_path = '/home/xiaoyao/data1/deep-Sep/model/diamond/db.dmnd'
    outfmt = '6 stitle qseqid sseqid pident length mismatch gapopen qstart qend sstart send sseq_gapped qseq_gapped evalue bitscore'
    # 调整线程和内存
    command = f"{program} blastp --db {db_path} -q {query} --out {out} --outfmt {outfmt} --threads 12 --masking 0 --faster --max-target-seqs 100 --id 20 -e 1e-2 -b 8 -c 1"
    os.system(command)

    logging.info('DIAMOND prediction ended on the dataset')

def match(df):
    all_seqs = []

    id_list = list(set(df['qseqid']))
    for i in id_list:
        indeed_df = df[df['qseqid'] == i]
        genomelist = set(indeed_df['genome'])

        if len(genomelist) < 2:
            continue
        else:
            UU_result = find_UU(indeed_df)
            if isinstance(UU_result, tuple):
                all_seqs.append(UU_result)
            else:
                UC_tup = find_UC(indeed_df)
                if isinstance(UC_tup, list):
                    all_seqs.append(UC_tup)

    return all_seqs

def find_UU(df):
    """
    U-U
    """
    for index, row in df.iterrows():
        stitle = row[0]
        qseqid = row[1]
        sseq = row[-5]
        qseq = row[-4]
        genome = row[-1]

        qseq_loca_U = [index for (index, value) in enumerate(qseq) if value == 'X']
        if qseq_loca_U:
            if any(qseq[i] == 'X' and sseq[i] == 'X' for i in qseq_loca_U):
                return qseqid, qseq, stitle, sseq
        else:
            continue

    return None

def find_UC(df):
    """
    U-C
    """
    candidate_seqs = []
    genomes = set()
    sseqs = set()
    for index, row in df[:10].iterrows():
        stitle = row[0]
        qseqid = row[1]
        sseq = row[-5]
        qseq = row[-4]
        genome = row[-1]

        qseq_loca_U = [index for (index, value) in enumerate(qseq) if value == 'X']
        if qseq_loca_U:
            if any(qseq[i] == 'X' and sseq[i] == 'C' for i in qseq_loca_U):
                if genome not in genomes and sseq not in sseqs:
                    candidate_seqs.append((qseqid, qseq))
                    genomes.add(genome)
                    sseqs.add(sseq)
        else:
            continue

    if len(candidate_seqs) >= 2:
        return candidate_seqs
    else:
        return None

def extract_pattern(title):
    if 'OS=' in title:
        pattern = r'OS=([^(\n]+?)(?=\s\(|\sOX=|$)'
    else:
        pattern = r"\[(.*?)\][^\[]*$"

    match = re.search(pattern, title)
    if match:
        return match.group(1)

    return None

def write(seqs, out_path, file_name):
    """
    将匹配结果写入 out_path/file_name
    - 单个 4-tuple: (qseqid, qseq, stitle, sseq)
    - list 内部也是 4-tuple
    - 2-tuple 仅有 (qseqid, qseq) 时，标记为 # no_hit
    """
    with open(os.path.join(out_path, file_name), "w") as f:
        if not seqs:      # 空列表
            f.write("None\n")
            return

        for item in seqs:
            # 单个 tuple
            if isinstance(item, tuple):
                if len(item) == 4:                  # 命中 ≥1
                    qseqid, qseq, stitle, sseq = item
                    f.write(f">{qseqid}\n")
                elif len(item) == 2:                # 无命中
                    qseqid, qseq = item
                    f.write(f">{qseqid} # no_hit\n")
                else:
                    # 意外长度，写日志或直接跳过
                    continue

            # list of tuples
            elif isinstance(item, list):
                for i in item:
                    if len(i) == 4:
                        qseqid, qseq, stitle, sseq = i
                        f.write(f">{qseqid}\n")
                    elif len(i) == 2:
                        qseqid, qseq = i
                        f.write(f">{qseqid} # no_hit\n")

            # 其它类型，忽略或记录
            else:
                continue


def analysis(result_output_dir, diamond_pred_results):
    with open(os.path.join(result_output_dir, 'preliminary_result-log.txt'), 'w') as f:
        df = pd.read_csv(diamond_pred_results, sep='\t',
                         names=['stitle', 'qseqid', 'sseqid', 'pident', 'length ', 'mismatch', 'gapopen ', 'qstart',
                                'qend', 'sstart', 'send', 'sseq', 'qseq', 'evalue', 'bitscore'])

        print('Find all queries: ' + str(len(set(df['qseqid']))), file=f, flush=True)
        print('All searched seqs num: ' + str(len(df)), file=f, flush=True)

        new_df = df.loc[:, :]
        new_df['genome'] = df['stitle'].apply(extract_pattern)
        new_df['genome'] = new_df['genome'].str.strip()

        print('\n' + '--------Candidates--------', file=f, flush=True)
        all_matched_seqs_tup = match(new_df)
        write(all_matched_seqs_tup, result_output_dir, 'preliminary_result.csv')

        print('Candidates: ' + str(len(all_matched_seqs_tup)), file=f, flush=True)

    return pd.read_csv(os.path.join(result_output_dir, 'preliminary_result.csv'), names=['qseqid'])

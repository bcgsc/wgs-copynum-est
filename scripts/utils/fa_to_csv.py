import argparse
import pandas as pd
import re

argparser = argparse.ArgumentParser(description='Write contig information in FASTA file to CSV')
argparser.add_argument('fasta_path', help='FASTA filepath')
argparser.add_argument('csv_output_path', help='Filepath for CSV output')
args = argparser.parse_args()

seqs = { 'ID': [], 'length': [], 'kmer_depth_sum': [], 'seq': [] }
fa_file = open(args.fasta_path)
for line in fa_file:
    if re.search('^>[0-9]', line):
        row = line[1:].split()
        seqs['ID'].append(row[0])
        seqs['length'].append(row[1])
        seqs['kmer_depth_sum'].append(row[2])
    else:
        seqs['seq'].append(line.strip())
seqs = pd.DataFrame(seqs)
seqs.set_index('ID', inplace=True)
seqs.to_csv(args.csv_output_path, index_label='ID')

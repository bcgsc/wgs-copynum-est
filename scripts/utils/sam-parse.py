import argparse
import csv
import re
import sys

ID_COL = 0
FLAGS_COL = 1
MAPQ_COL = 4
CIGAR_COL = 5
SEQ_COL = 9

def compute_gc_content(seq):
    gc_count = 0
    for b in seq:
        if b == 'G' or b == 'C':
            gc_count += 1
    return (gc_count / len(seq))

def init_seq_dict(row):
    return { 'ID': int(row[ID_COL]), 'Matches': 0, 'Clipped': 0, 'MAPQ sum': 0, 'GC content': compute_gc_content(row[SEQ_COL]) }

def update_match_info(row, seq_dict):
    if is_alnmt_clipped(row[CIGAR_COL]):
        seq_dict['Clipped'] += 1
    else:
        seq_dict['Matches'] += 1
        seq_dict['MAPQ sum'] += int(row[MAPQ_COL])

def is_mapped(flags):
    if flags >= 512:
        return False
    mod = 256
    while flags > 0:
        if flags >= mod:
            if mod == 4:
                return False
            flags -= mod
        mod /= 2
    return True

def is_alnmt_clipped(cigar):
    hsplit = cigar.split('H')
    if len(hsplit) == 1:
        return (len(cigar.split('S')) > 1)
    return True


argparser = argparse.ArgumentParser(description="Parse SAM file output by BWA-MEM alignment of sequences/unitigs in FASTA file to [mutated] reference")
argparser.add_argument("samfilename", type=str, help="BWA-MEM output SAM file")
argparser.add_argument("outfilename", type=str, help="File name for sequence/unitig reference alignment counts output")
args = argparser.parse_args()

csv.field_size_limit(sys.maxsize)

with open(args.samfilename, newline='') as samfile:
    reader = csv.reader(samfile, delimiter='\t')
    with open(args.outfilename, 'w', newline='') as outfile:
        writer = csv.DictWriter(outfile, delimiter='\t', fieldnames=['ID', 'Matches', 'Clipped', 'MAPQ sum', 'GC content'])
        writer.writeheader()
        row = next(reader)
        seq_dict = init_seq_dict(row)
        if is_mapped(int(row[FLAGS_COL])):
            update_match_info(row, seq_dict)
        i = 1
        for row in reader:
            if int(row[ID_COL]) == seq_dict['ID']:
                if is_mapped(int(row[FLAGS_COL])):
                    update_match_info(row, seq_dict)
            else:
                writer.writerow(seq_dict)
                seq_dict = init_seq_dict(row)
                if is_mapped(int(row[FLAGS_COL])):
                    update_match_info(row, seq_dict)
        writer.writerow(seq_dict)


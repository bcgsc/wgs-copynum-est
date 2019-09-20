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
    return {'ID': int(row[ID_COL]), 'Mapped': is_mapped(int(row[FLAGS_COL])), 'Matches': 0, 'Others': 0, 'MAPQ sum': 0, 'GC content': compute_gc_content(row[SEQ_COL]) }

def update_match_info(row, seq_dict):
    if is_alnmt_hard_clipped(row[CIGAR_COL]):
        seq_dict['Others'] += 1
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

def is_match(cigar):
    if re.search('^[0-9]+M$', cigar):
        return True
    return False

def is_alnmt_hard_clipped(cigar):
    if re.search('H', cigar):
        return True
    return False


argparser = argparse.ArgumentParser(description="Parse SAM file output by BWA-MEM alignment of sequences/unitigs in FASTA file to [mutated] reference")
argparser.add_argument("samfilename", type=str, help="BWA-MEM output SAM file")
argparser.add_argument("outfilename", type=str, help="File name for sequence/unitig reference alignment counts output")
argparser.add_argument("errorfilename", type=str, help="Error log file name")
args = argparser.parse_args()

csv.field_size_limit(sys.maxsize)

with open(args.samfilename, newline='') as samfile:
    reader = csv.reader(samfile, delimiter='\t')
    error_seqs = []
    with open(args.outfilename, 'w', newline='') as outfile:
        writer = csv.DictWriter(outfile, delimiter='\t', fieldnames=['ID', 'Mapped', 'Matches', 'Others', 'MAPQ sum', 'GC content'])
        writer.writeheader()
        row = next(reader)
        seq_dict = init_seq_dict(row)
        if seq_dict['Mapped']:
            update_match_info(row, seq_dict)
        i = 1
        for row in reader:
            if int(row[ID_COL]) == seq_dict['ID']:
                if is_mapped(int(row[FLAGS_COL])):
                    if seq_dict['Mapped']:
                        update_match_info(row, seq_dict)
                    elif error_seqs[-1] != seq_dict['ID']: # if listed as unmapped and then mapped...
                        error_seqs.append(seq_dict['ID'])
            else:
                writer.writerow(seq_dict)
                seq_dict = init_seq_dict(row)
                if seq_dict['Mapped']:
                    update_match_info(row, seq_dict)
        writer.writerow(seq_dict)
    with open(args.errorfilename, 'w', newline='') as errorfile:
        writer = csv.writer(errorfile)
        for id in error_seqs:
            writer.writerow([id])


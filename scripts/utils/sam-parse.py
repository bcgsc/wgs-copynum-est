import argparse
import csv
import re
import sys

ID_COL = 0
FLAGS_COL = 1
MAPQ_COL = 4
CIGAR_COL = 5
SEQ_COL = 9

ID_COLNAME = 'ID'
MATCHES_COLNAME = 'Matches'
CLIPPED_COLNAME ='Clipped'
MAPQ_SUM_COLNAME = 'MAPQ sum'
GC_CONTENT_COLNAME = 'GC content'

def compute_gc_content(seq):
    gc_count = 0
    for b in seq:
        if b == 'G' or b == 'C':
            gc_count += 1
    return (gc_count / len(seq))

def init_seq_dict(row):
    return { ID_COLNAME: int(row[ID_COL]), MATCHES_COLNAME: 0, CLIPPED_COLNAME: 0, MAPQ_SUM_COLNAME: 0, GC_CONTENT_COLNAME: compute_gc_content(row[SEQ_COL]) }

def update_match_info(row, seq_dict):
    if is_alnmt_clipped(row[CIGAR_COL]):
        seq_dict[CLIPPED_COLNAME] += 1
    else:
        seq_dict[MATCHES_COLNAME] += 1
        seq_dict[MAPQ_SUM_COLNAME] += int(row[MAPQ_COL])

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
        writer = csv.DictWriter(outfile, delimiter='\t', fieldnames=[ID_COLNAME, MATCHES_COLNAME, CLIPPED_COLNAME, MAPQ_SUM_COLNAME, GC_CONTENT_COLNAME])
        writer.writeheader()
        row = next(reader)
        seq_dict = init_seq_dict(row)
        if is_mapped(int(row[FLAGS_COL])):
            update_match_info(row, seq_dict)
        i = 1
        for row in reader:
            if int(row[ID_COL]) == seq_dict[ID_COLNAME]:
                if is_mapped(int(row[FLAGS_COL])):
                    update_match_info(row, seq_dict)
            else:
                writer.writerow(seq_dict)
                seq_dict = init_seq_dict(row)
                if is_mapped(int(row[FLAGS_COL])):
                    update_match_info(row, seq_dict)
        writer.writerow(seq_dict)


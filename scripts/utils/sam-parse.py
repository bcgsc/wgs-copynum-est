import argparse
import csv
import re
import sys

ID_COL = 0
FLAGS_COL = 1
MAPQ_COL = 4
CIGAR_COL = 5
SEQ_COL = 9
NM_COL = 11

ID_COLNAME = 'ID'
LENGTH_COLNAME = 'Length'
MATCHES_COLNAME = 'Matches'
CLIPPED_COLNAME ='Clipped'
MAPQ_SUM_COLNAME = 'MAPQ sum'
GC_CONTENT_COLNAME = 'GC content'
EDIT_DIST_COLNAME = 'Edit distances'

def compute_gc_content(seq):
    gc_count = 0.0
    for b in seq:
        if b == 'G' or b == 'C':
            gc_count += 1
    return (gc_count / len(seq))

def get_edit_distance(nm_col):
    return int(nm_col.split(':')[-1])

def init_seq_dict(row):
    return { ID_COLNAME: int(row[ID_COL]), LENGTH_COLNAME: len(row[SEQ_COL]), MATCHES_COLNAME: 0, CLIPPED_COLNAME: 0,
        MAPQ_SUM_COLNAME: 0, GC_CONTENT_COLNAME: compute_gc_content(row[SEQ_COL]), EDIT_DIST_COLNAME: {} }

def update_match_info(row, seq_dict):
    if is_alnmt_clipped(row[CIGAR_COL]):
        seq_dict[CLIPPED_COLNAME] += 1
    else:
        seq_dict[MATCHES_COLNAME] += 1
        seq_dict[MAPQ_SUM_COLNAME] += int(row[MAPQ_COL])
        edit_distance = get_edit_distance(row[NM_COL])
        if edit_distance not in seq_dict[EDIT_DIST_COLNAME]:
            seq_dict[EDIT_DIST_COLNAME][edit_distance] = 0
        seq_dict[EDIT_DIST_COLNAME][edit_distance] += 1

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

samfile = open(args.samfilename, newline='')
row = samfile.readline()
while not(re.match('^@SQ', row)):
    row = samfile.readline()
while re.match('^@SQ', row):
    row = samfile.readline()
outfile = open(args.outfilename, 'w', newline='')
writer = csv.DictWriter(outfile, delimiter='\t', fieldnames=[ID_COLNAME, LENGTH_COLNAME, MATCHES_COLNAME, CLIPPED_COLNAME, MAPQ_SUM_COLNAME, GC_CONTENT_COLNAME, EDIT_DIST_COLNAME])
writer.writeheader()
reader = csv.reader(samfile, delimiter='\t')
row = next(reader)
seq_dict = init_seq_dict(row)
if is_mapped(int(row[FLAGS_COL])):
    update_match_info(row, seq_dict)
for row in reader:
    if not(re.match('^\d+$', row[0])):
        break
    if int(row[ID_COL]) == seq_dict[ID_COLNAME]:
        if is_mapped(int(row[FLAGS_COL])):
            update_match_info(row, seq_dict)
    else:
        seq_dict[EDIT_DIST_COLNAME] = str(seq_dict[EDIT_DIST_COLNAME])[1:-1].replace(' ', '')
        writer.writerow(seq_dict)
        seq_dict = init_seq_dict(row)
        if is_mapped(int(row[FLAGS_COL])):
            update_match_info(row, seq_dict)
seq_dict[EDIT_DIST_COLNAME] = str(seq_dict[EDIT_DIST_COLNAME])[1:-1].replace(' ', '')
writer.writerow(seq_dict)
samfile.close()

import argparse
import csv
import re
import sys


ID_COL = 0
FLAGS_COL = 1
CHROM_COL = 2
POS_COL = 3
MAPQ_COL = 4
CIGAR_COL = 5
SEQ_COL = 9
NM_COL = 11
MD_COL = 12

ID_COLNAME = 'ID'
LENGTH_COLNAME = 'length'
RV_COMPLEMENT_COLNAME = 'rv_complement'
CHROM_COLNAME = 'chromosome'
START_POS_COLNAME = 'start_pos'
END_POS_COLNAME = 'end_pos'
MAPQ_COLNAME = 'MAPQ'
CIGAR_COLNAME = 'CIGAR'
MD_COLNAME = 'MD'
EDIT_DIST_COLNAME = 'edit_distance'
TOTAL_EDIT_DIST_COLNAME = 'total_edit_dist'
SEQ_IDENTITY_COLNAME = 'sequence_identity'

CONTINUE_RE = re.compile('^\d+$')
BAD_CIGAR_RE = re.compile('[*NP=X]')
CIGAR_RE = re.compile('\d+[MIDSH]')

# Primarily checks for mapping flag 4, but also a CIGAR string I can't handle, just in case
# Partly repeated in the next function, but to hell with code hygiene at this point
def is_ok(flags, cigar):
    if has_bad_cigar(cigar):
        return False
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

def reverse_complemented(flags):
    mod = 256
    while flags > 8:
        if flags >= mod:
            if mod == 16:
                return True
            flags -= mod
        mod /= 2
    return False

def get_cigar_stubs(cigar):
    return CIGAR_RE.findall(cigar)

def compute_ref_end_pos(pos, stubs):
    for stub in stubs:
        if (stub[-1] == 'M') or (stub[-1] == 'D'):
            pos += int(stub[:-1])
    return pos - 1

def parse_MD(mdfield):
    return mdfield.split(':')[2]

def get_edit_distance(nm_col):
    return int(nm_col.split(':')[-1])

def get_clip_len(clip_stub):
    if (clip_stub[-1] == 'S') or (clip_stub[-1] == 'H'):
        return int(clip_stub[:-1])
    return 0

def get_total_edit_dist(cigar_stubs, edit_dist):
    return get_clip_len(cigar_stubs[0]) + edit_dist + get_clip_len(cigar_stubs[-1])

def get_seq_identity(seq_len, total_edit_dist):
    return (seq_len - total_edit_dist) * 1.0 / seq_len

def init_aln_dict(row):
    return { ID_COLNAME: int(row[ID_COL]), LENGTH_COLNAME: len(row[SEQ_COL]) }

def update_aln_dict(aln_dict, row, ok):
    aln_dict[CHROM_COLNAME] = row[CHROM_COL]
    aln_dict[START_POS_COLNAME] = row[POS_COL]
    aln_dict[MAPQ_COLNAME] = row[MAPQ_COL]
    aln_dict[CIGAR_COLNAME] = row[CIGAR_COL]
    if ok:
        aln_dict[RV_COMPLEMENT_COLNAME] = reverse_complemented(int(row[FLAGS_COL]))
        stubs = get_cigar_stubs(row[CIGAR_COL])
        edit_dist = get_edit_distance(row[NM_COL])
        total_edit_dist = get_total_edit_dist(stubs, edit_dist)
        aln_dict[END_POS_COLNAME] = compute_ref_end_pos(int(row[POS_COL]), stubs)
        aln_dict[MD_COLNAME] = parse_MD(row[MD_COL])
        aln_dict[EDIT_DIST_COLNAME] = edit_dist
        aln_dict[TOTAL_EDIT_DIST_COLNAME] = total_edit_dist
        aln_dict[SEQ_IDENTITY_COLNAME] = get_seq_identity(aln_dict[LENGTH_COLNAME], total_edit_dist)
    else:
        aln_dict[RV_COMPLEMENT_COLNAME] = None
        aln_dict[END_POS_COLNAME] = None
        aln_dict[MD_COLNAME] = None
        aln_dict[EDIT_DIST_COLNAME] = None
        aln_dict[TOTAL_EDIT_DIST_COLNAME] = None
        aln_dict[SEQ_IDENTITY_COLNAME] = None

# Probably should never be false in my usage, but...
def has_bad_cigar(cigar):
    return BAD_CIGAR_RE.search(cigar)


argparser = argparse.ArgumentParser(description="Parse SAM file output by BWA-MEM alignment of real genome assembly sequences in FASTA file to reference")
argparser.add_argument("samfilename", type=str, help="BWA-MEM output SAM file")
argparser.add_argument("outfilename", type=str, help="File name for sequence reference alignment summary data output")
args = argparser.parse_args()

csv.field_size_limit(sys.maxsize)

samfile = open(args.samfilename, newline='')
row = samfile.readline()
while not(re.match('^@SQ', row)):
    row = samfile.readline()
while re.match('^@SQ', row):
    row = samfile.readline()
outfile = open(args.outfilename, 'w', newline='')
writer = csv.DictWriter(outfile, delimiter='\t', fieldnames=[ID_COLNAME, LENGTH_COLNAME, RV_COMPLEMENT_COLNAME, CHROM_COLNAME,
  START_POS_COLNAME, END_POS_COLNAME, MAPQ_COLNAME, CIGAR_COLNAME, MD_COLNAME, EDIT_DIST_COLNAME, TOTAL_EDIT_DIST_COLNAME, SEQ_IDENTITY_COLNAME])
writer.writeheader()
reader = csv.reader(samfile, delimiter='\t')
row = next(reader)
aln_dict = init_aln_dict(row)
update_aln_dict(aln_dict, row, is_ok(int(row[FLAGS_COL]), row[CIGAR_COL]))
writer.writerow(aln_dict)
for row in reader:
    if not(CONTINUE_RE.match(row[0])):
        break
    if int(row[ID_COL]) != aln_dict[ID_COLNAME]:
        aln_dict = init_aln_dict(row)
    update_aln_dict(aln_dict, row, is_ok(int(row[FLAGS_COL]), row[CIGAR_COL]))
    writer.writerow(aln_dict)
samfile.close()

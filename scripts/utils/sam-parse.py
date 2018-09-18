import csv
import re
import sys

SAMFILE = sys.argv[1]
OUTFILE = sys.argv[2]
ERRORFILE = sys.argv[3]
MISSINGFILE = sys.argv[4]
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
    return {'ID': int(row[ID_COL]), 'Mapped': is_mapped(int(row[FLAGS_COL])), 'Matches': 0, 'Others': 0, 'Other CIGARs': '', 'MAPQ (unique match only)': 0, 'GC content': compute_gc_content(row[SEQ_COL]) }

def update_match_info(row, seq_dict, other_cigars):
    if is_match_lax(row[CIGAR_COL]):
        seq_dict['Matches'] += 1
        seq_dict['MAPQ (unique match only)'] += int(row[MAPQ_COL])
    else:
        seq_dict['Others'] += 1
        other_cigars.append(row[CIGAR_COL])

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

def is_match_lax(cigar):
    if re.search('H', cigar):
        return False
    return True

def finalise(seq_dict, other_cigars):
    seq_dict['Other CIGARs'] = ','.join(other_cigars)


csv.field_size_limit(sys.maxsize)

with open(SAMFILE, newline='') as samfile:
    reader = csv.reader(samfile, delimiter='\t')
    error_seqs = []
    missing_seqs = []
    with open(OUTFILE, 'w', newline='') as outfile:
        writer = csv.DictWriter(outfile, delimiter='\t', fieldnames=['ID', 'Mapped', 'Matches', 'Others', 'Other CIGARs', 'MAPQ (unique match only)', 'GC content'])
        writer.writeheader()
        row = next(reader)
        seq_dict = init_seq_dict(row)
        other_cigars = []
        if seq_dict['Mapped']:
            update_match_info(row, seq_dict, other_cigars)
        i = 1
        for row in reader:
            if int(row[ID_COL]) == seq_dict['ID']:
                if is_mapped(int(row[FLAGS_COL])):
                    if seq_dict['Mapped']:
                        update_match_info(row, seq_dict, other_cigars)
                    elif error_seqs[-1] != seq_dict['ID']: # if listed as unmapped and then mapped...
                        error_seqs.append(seq_dict['ID'])
            else:
                if int(row[ID_COL]) == seq_dict['ID'] + 1:
                    finalise(seq_dict, other_cigars)
                    writer.writerow(seq_dict)
                    seq_dict = init_seq_dict(row)
                    other_cigars = []
                    if seq_dict['Mapped']:
                        update_match_info(row, seq_dict, other_cigars)
                else:
                    missing_seqs.extend(range(seq_dict['ID'] + 1, int(row[0])))
            i += 1
        finalise(seq_dict, other_cigars)
        writer.writerow(seq_dict)
    with open(ERRORFILE, 'w', newline='') as errorfile:
        writer = csv.writer(errorfile)
        for id in error_seqs:
            writer.writerow([id])
    with open(MISSINGFILE, 'w', newline='') as missingfile:
        writer = csv.writer(missingfile)
        for id in missing_seqs:
            writer.writerow([id])


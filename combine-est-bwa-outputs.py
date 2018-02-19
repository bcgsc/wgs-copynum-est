import csv
import numpy as np
import sys

def write_table(tfile, table):
    writer = csv.writer(tfile)
    writer.writerow(['Est/Aln', '0', '1', '2', '3', '4', '5+'])
    for i in range(1,5):
        row = [i]
        row.extend(table[i])
        writer.writerow(row)
    row = ['5+']
    row.extend(table[5])
    writer.writerow(row)

EST_OUTPUT = sys.argv[1]
BWA_PARSE_OUTPUT = sys.argv[2]
KMER = int(sys.argv[3])

# REPLACE MAGIC NUMBER
seqs = np.zeros(129026, dtype=[('ID', np.uint64), ('kmers', np.uint64), ('avg_depth', np.float64), ('est_label', np.uint64), ('aln_match_count', np.uint64), ('aln_other_count', np.uint64)])
table = np.zeros((6,6), dtype=np.uint)
table_lt100 = np.zeros((6,6), dtype=np.uint)
table_lt1000 = np.zeros((6,6), dtype=np.uint)
table_lt10000 = np.zeros((6,6), dtype=np.uint)
table_10000plus = np.zeros((6,6), dtype=np.uint)

with open(EST_OUTPUT, newline='') as estfile:
    reader = csv.reader(estfile)
    next(reader)
    for row in reader:
        #print(row[0])
        #print(seqs[0])
        #print(seqs[row[0]])
        seqID = int(row[0])
        seqs[seqID]['ID'] = seqID
        seqs[seqID]['kmers'] = int(row[1])
        seqs[seqID]['avg_depth'] = float(row[2])
        seqs[seqID]['est_label'] = int(row[3]) + 1

with open(BWA_PARSE_OUTPUT, newline='') as alnfile:
    reader = csv.reader(alnfile, delimiter='\t')
    next(reader)
    for row in reader:
        seqID = int(row[0])
        seqs[seqID]['aln_match_count'] = int(row[2])
        seqs[seqID]['aln_other_count'] = int(row[3])

with open('seq-est-and-aln.csv', 'w', newline='') as seqfile:
    writer = csv.writer(seqfile)
    writer.writerow(['ID', 'Length (in kmers)', 'Average depth', 'Est. likeliest label', 'Ref. matches', 'Ref. other matches'])
    for i in range(len(seqs)):
        writer.writerow(list(seqs[i]))

for i in range(len(seqs)):
    rowidx = 5
    colidx = 5
    if seqs[i]['est_label'] < 5:
        rowidx = seqs[i]['est_label']
    if seqs[i]['aln_match_count'] < 5:
        colidx = seqs[i]['aln_match_count']
    table[rowidx][colidx] += 1
    if seqs[i]['kmers'] + KMER - 1 < 100:
        table_lt100[rowidx][colidx] += 1
    elif seqs[i]['kmers'] + KMER - 1 < 1000:
        table_lt1000[rowidx][colidx] += 1
    elif seqs[i]['kmers'] + KMER - 1 < 10000:
        table_lt10000[rowidx][colidx] += 1
    else:
        table_10000plus[rowidx][colidx] += 1

with open('est-aln-counts.csv', 'w') as tfile:
    write_table(tfile, table)
    
with open('est-aln-counts-lt100.csv', 'w') as tfile:
    write_table(tfile, table_lt100)

with open('est-aln-counts-lt1000.csv', 'w') as tfile:
    write_table(tfile, table_lt1000)

with open('est-aln-counts-lt10000.csv', 'w') as tfile:
    write_table(tfile, table_lt10000)

with open('est-aln-counts-10000plus.csv', 'w') as tfile:
    write_table(tfile, table_10000plus)

import argparse
import re

argparser = argparse.ArgumentParser(description="Write data of interest in FASTG file containing SPAdes contigs output by Unicycler to FASTA file.")
argparser.add_argument("fastg_filename", type=str, help="Input FASTG file name")
argparser.add_argument("fasta_filename", type=str, help="Output FASTA file name")
args = argparser.parse_args()

fastg_file = open(args.fastg_filename, newline='')
outfile = open(args.fasta_filename, 'w', newline='')
row = fastg_file.readline()
while re.search('^S', row):
    cols = row.split()
    outfile.write('>' + cols[1] + ' ' + cols[3].split(':')[2] + ' ' + cols[4].split(':')[2] + '\n')
    outfile.write(cols[2])
    outfile.write('\n')
    row = fastg_file.readline()

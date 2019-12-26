import argparse
import glob
import gzip
import re


DESCRIPTION = 'Concatenate chromosome-level reference genome FASTA files into a single consolidated FASTA file with 2 lines (name & sequence) per chromosome'
PREFIX_HELP = 'Initial prefix shared across all single-chromosome FASTA files, with possible intervening characters before chromosome number and file suffix'
argparser = argparse.ArgumentParser(description=DESCRIPTION)
argparser.add_argument('prefix', type=str, help=PREFIX_HELP)
args = argparser.parse_args()

files = glob.glob(args.prefix + '*.fa*')
if len(files) > 1:
  chromosome_ids = list(map(lambda f: re.match('.+\.(\d{1,2}|W|X|Y|Z)\.fa.*', f).group(1), files))
  chromosome_ids.sort()
  input_filename_parts = re.match('(.+)\.(\d{1,2}|W|X|Y|Z)\.(fa.*)', files[0])
  input_filename_prefix, input_filename_suffix = input_filename_parts.group(1), input_filename_parts.group(3)
  filenames = map(lambda i: input_filename_prefix + '.' + i + '.' + input_filename_suffix, chromosome_ids)
else:
  filenames = files

outfile = open(args.prefix + '.fa', 'w')
for f in filenames:
  if re.match('.*\.fa\.gz', f):
    infile = gzip.open(f, mode='rt')
  else:
    infile = open(f)
  line = infile.readline()
  outfile.write(line)
  line = infile.readline()
  while line:
    outfile.write(line.rstrip())
    line = infile.readline()
  outfile.write('\n')
  infile.close()
outfile.close()

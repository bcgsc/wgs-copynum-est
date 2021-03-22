import argparse
import csv
import gzip
import re
import sys


CHROM_COL = 0
POS_COL = 1
REF_COL = 3
ALT_COL = 4
QUAL_COL = 5
INFO_COL = 7
SAMPLE_COL = 9

CHROM_COLNAME = 'chromosome'
POS_COLNAME = 'position'
REF_COLNAME = 'ref_base'
ALT_COLNAME = 'alt_base'
QUAL_COLNAME = 'alt_call_qual'
MONO_ALLELIC_VAR_COLNAME = 'monoallelic_variant' # boolean
AA_LIK_COLNAME = 'AA_phred_rel_lik'
AB_LIK_COLNAME = 'AB_phred_rel_lik'
BB_LIK_COLNAME = 'BB_phred_rel_lik'
DP_COLNAME = 'avg_read_depth'
REF_DP_COLNAME = 'ref_allele_depth'
ALT_DP_COLNAME = 'alt_allele_depth'
MQ_COLNAME = 'avg_map_qual'
MQ0F_COLNAME = 'fraction_mapq_0'

def row_to_dict(row):
    sample_fields = row[SAMPLE_COL].split(':')
    phredsc_likelihoods = sample_fields[1].split(',')
    allelic_depths = sample_fields[3].split(',')
    info_pairs = list(map(lambda p: p.split('='), row[INFO_COL].split(';')))
    if len(info_pairs[0]) == 1:
        info_pairs[0] = [info_pairs[0][0], '.']
    info_dict = { p[0]: p[1] for p in info_pairs }
    return { CHROM_COLNAME: row[CHROM_COL], POS_COLNAME: row[POS_COL], REF_COLNAME: row[REF_COL], ALT_COLNAME: row[ALT_COL], QUAL_COLNAME: row[QUAL_COL],
        MONO_ALLELIC_VAR_COLNAME: int(sample_fields[0][0] == '1'), AA_LIK_COLNAME: phredsc_likelihoods[0], AB_LIK_COLNAME: phredsc_likelihoods[1], BB_LIK_COLNAME: phredsc_likelihoods[2],
        DP_COLNAME: sample_fields[2], REF_DP_COLNAME: allelic_depths[0], ALT_DP_COLNAME: allelic_depths[1], MQ_COLNAME: info_dict['MQ'], MQ0F_COLNAME: info_dict['MQ0F'] }


argparser = argparse.ArgumentParser(description="Parse VCF file output by bcftools mpileup and call based on BWA-MEM alignment of real genomic reads to reference")
argparser.add_argument("vcffilepath", type=str, help="VCF file path")
argparser.add_argument("outfilepath", type=str, help="File path for parsed output")
args = argparser.parse_args()

csv.field_size_limit(sys.maxsize)

vcffile = gzip.open(args.vcffilepath, 'rt')
reader = csv.reader(vcffile, delimiter='\t')
for row in reader:
    if not re.match('^#', row[0]):
        break
outfile = open(args.outfilepath, 'w', newline='')
writer = csv.DictWriter(outfile, delimiter='\t', fieldnames=[CHROM_COLNAME, POS_COLNAME, REF_COLNAME, ALT_COLNAME, QUAL_COLNAME,
  MONO_ALLELIC_VAR_COLNAME, AA_LIK_COLNAME, AB_LIK_COLNAME, BB_LIK_COLNAME, DP_COLNAME, REF_DP_COLNAME, ALT_DP_COLNAME, MQ_COLNAME, MQ0F_COLNAME])
writer.writeheader()
writer.writerow(row_to_dict(row))
for row in reader:
    writer.writerow(row_to_dict(row))
vcffile.close()

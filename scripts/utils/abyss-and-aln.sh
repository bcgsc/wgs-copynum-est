#!/bin/bash

set -x

k=$1; shift
abyss_t=$1; shift
reads1=$1; shift
reads2=$1; shift
name=$1; shift
bwa_t=$1; shift
refpath=$1; shift

mkdir -p abyss-out
cd abyss-out
${WGS_COPYNUM_EST_HOME}/scripts/run-abyss.sh $k $abyss_t ../$reads1 ../$reads2 $name

cd ..
mkdir -p aln

# Run BWA-mem to align unitigs to reference; output to SAM file.
# Recommended value for -t: 1. Otherwise output is interleaved and complicates parsing.
# Other args to BWA-MEM, e.g.: -w 5 -d 50 -c 10000
bwa mem -a -k $k -t $bwa_t $@ $refpath abyss-out/${name}-2.fa > aln/${name}_aln.sam

# Parse SAM; write to <ref_aln_counts_csv> (substitute perfect reads SAM file for simulated dataset if appropriate)
python ${WGS_COPYNUM_EST_HOME}/scripts/utils/sam_parse.py aln/${name}_aln.sam aln/${name}_aln_parsed.tsv

set +x

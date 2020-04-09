#!/bin/bash

set -x

k=$1; shift
log_last_multiplicity_line=$1; shift
log_multiplicity_line_count=$1; shift
outdir=$1; shift
bwa_t=$1; shift
refpath=$1; shift

mkdir -p $outdir
head -$log_last_multiplicity_line unicycler.log | tail -$log_multiplicity_line_count > $outdir/graph_multiplicities.txt
python ${WGS_COPYNUM_EST_HOME}/scripts/unicycler/fastg2fasta.py 001_best_spades_graph.gfa $outdir/spades_contigs.fasta
# Recommended value for -t: 1. Otherwise output is interleaved and complicates parsing.
cd $outdir
mkdir -p unicycler-cpnum-results
python ${WGS_COPYNUM_EST_HOME}/scripts/unicycler/get_multiplicities.py spades_contigs.fasta graph_multiplicities.txt unicycler-cpnum-results
bwa mem -a -k $k -t $bwa_t $@ ../$refpath spades_contigs.fasta > contigs_aln.sam

python ${WGS_COPYNUM_EST_HOME}/scripts/utils/sam-parse.py contigs_aln.sam contigs_aln-counts.tsv

mkdir -p est-cpnum-results
${WGS_COPYNUM_EST_HOME}/scripts/unicycler/aln-and-est.sh $k est-cpnum-results

cd unicycler-cpnum-results
python ${WGS_COPYNUM_EST_HOME}/scripts/results/general/combine-est-bwa-outputs.py --collapse_highest_est_cpnums sequence-labels.csv ../contigs_aln-counts.tsv ../est-cpnum-results/length_gp_stats.csv
${WGS_COPYNUM_EST_HOME}/scripts/utils/compute-stats-loop.sh

set +x

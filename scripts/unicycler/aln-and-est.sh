#!/bin/bash

set -x

k=$1
outdir=$2

# Run estimator
python ${WGS_COPYNUM_EST_HOME}/scripts/est.py --haploid --per_unitig_mean_depth_given spades_contigs.fasta $k $outdir
# Combine estimation and reference alignment data
cd $outdir
python ${WGS_COPYNUM_EST_HOME}/scripts/results/general/combine-est-bwa-outputs.py sequence-labels.csv ../contigs_aln-counts.tsv length_gp_stats.csv
# Compute summary stats
${WGS_COPYNUM_EST_HOME}/scripts/utils/compute-stats-loop.sh

set +x

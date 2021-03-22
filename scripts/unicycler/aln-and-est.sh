#!/bin/bash

set -x

k=$1
outdir=$2

# Run estimator
python ${WGS_COPYNUM_EST_HOME}/scripts/est.py --haploid --per_unitig_mean_depth_given spades_contigs.fasta $k $outdir
# Combine estimation and reference alignment data
cd $outdir
python ${WGS_COPYNUM_EST_HOME}/scripts/results/general/combine_est_bwa_outputs.py sequence-labels.csv ../contigs_aln_parsed.tsv length_gp_stats.csv
# Compute summary stats
${WGS_COPYNUM_EST_HOME}/scripts/utils/results/compute-stats-loop.sh

set +x

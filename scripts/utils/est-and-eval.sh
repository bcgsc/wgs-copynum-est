#!/bin/bash

haploid=""
half=""
longest_seqs_peak_expected_cpnum=""
use_length_strata=""
lax_best_alnmt=""
no_seq_identity_matching=""
seq_identity_matching=""
seq_identity_threshold=""

while getopts "Hhc:oleit:" opt; do
  case $opt in
    H)
      haploid="--haploid"
      ;;
    h)
      half="--half"
      ;;
    c)
      longest_seqs_peak_expected_cpnum="$OPTARG"
      ;;
    o)
      use_length_strata="--use_length_strata o"
      ;;
    l)
      lax_best_alnmt="--lax_best_alnmt"
      ;;
    e)
      no_seq_identity_matching="--no_seq_identity_matching"
      ;;
    i)
      seq_identity_matching="--seq_identity_matching"
      ;;
    t)
      seq_identity_threshold="--seq_identity_threshold $OPTARG"
      ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      ;;
  esac
done

shift $((OPTIND-1))
name=$1
k=$2

set -x

# Run estimator
mkdir -p results
python ${WGS_COPYNUM_EST_HOME}/scripts/est.py $haploid $half abyss-out/${name}-2.fa $k results $longest_seqs_peak_expected_cpnum

# Combine estimation and reference alignment data
cd results
mkdir -p full
mkdir -p summary
mv sequence-labels.csv full/
mv length_gp_stats.csv copynumber_params.csv summary/
python ${WGS_COPYNUM_EST_HOME}/scripts/results/general/combine_est_bwa_outputs.py $haploid $use_length_strata $lax_best_alnmt $no_seq_identity_matching $seq_identity_matching $seq_identity_threshold full/sequence-labels.csv ../aln/${name}_aln_parsed.tsv summary/length_gp_stats.csv

mv seq-est-and-aln.csv full/
mv aln-est* summary/

# Compute summary stats
cd summary
${WGS_COPYNUM_EST_HOME}/scripts/utils/results/compute-stats-loop.sh

mkdir -p aln-est_counts
mv aln-est_counts*csv aln-est_counts/
mkdir -p counts
mv counts_*csv counts/
mkdir -p summary-stats
mv summary-stats*csv summary-stats/

set +x

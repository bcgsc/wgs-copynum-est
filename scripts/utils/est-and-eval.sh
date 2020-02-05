#!/bin/bash

haploid=""
half=""
use_length_strata=""
use_oom_len_gps=""
ideal=""

while getopts "Hhoi" opt; do
  case $opt in
    H)
      haploid="--haploid"
      ;;
    h)
      half="--half"
      ;;
    o)
      use_length_strata="--use_length_strata o"
      use_oom_len_gps="--use_oom_len_gps"
      ;;
    i)
      ideal="ideal"
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
python ${WGS_COPYNUM_EST_HOME}/scripts/est.py $haploid $half abyss-out/${name}-2.fa $k results

# Combine estimation and reference alignment data
cd results
mkdir -p full
mkdir -p summary
mv sequence-labels.csv full/
mv length_gp_stats.csv copynumber_params.csv summary/
python ${WGS_COPYNUM_EST_HOME}/scripts/results/general/combine-est-bwa-outputs.py $haploid $use_length_strata full/sequence-labels.csv ../aln/${name}_aln-counts.tsv summary/length_gp_stats.csv

mv seq-est-and-aln.csv full/
mv aln-est* summary/

# Compute summary stats
cd summary
${WGS_COPYNUM_EST_HOME}/scripts/utils/compute-stats-loop.sh

mkdir -p aln-est_counts
mv aln-est_counts*csv aln-est_counts/
mkdir -p counts
mv counts_*csv counts/
mkdir -p summary-stats
mv summary-stats*csv summary-stats/

# Create plots; compute ideal (best-possible) classifier performance summary statistics.
cd ..
mkdir -p plots
mkdir -p plots/observed
if [[ "${ideal}" == "ideal" ]]; then
  mkdir -p ideal
fi
python ${WGS_COPYNUM_EST_HOME}/scripts/results/general/component-densities_ideal-summary-stats.py $use_oom_len_gps full/seq-est-and-aln.csv summary/length_gp_stats.csv summary/copynumber_params.csv plots/observed component-densities $ideal
if [ -z "$(ls -A plots/observed)" ]; then
  rm -rf plots/
  if [ -d ideal ]; then
    rmdir ideal
  fi
else
  mkdir -p summary/summary-stats/compare_ideal/
  # Compute classifier-to-ideal performance summary statistics ratios
  ${WGS_COPYNUM_EST_HOME}/scripts/utils/compare_est-ideal-stats_loop.sh
fi

set +x

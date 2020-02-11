#!/bin/bash

k=$1
name=$2
max=$3
ntcard_prefix=$4
out_folder=$5
gs_results_folder=$6
est_results_folder=$7

if [ "$max" -lt 3 ]; then
  max=""
else
  max="--max_cpnum_3"
fi

set -x

python ${WGS_COPYNUM_EST_HOME}/scripts/genomescope/cpnums-from-param-estimates.py $k $max k$k/abyss-out/$name-2.fa ${ntcard_prefix}_k${k}.hist $out_folder $gs_results_folder
cd $out_folder/$gs_results_folder
python ${WGS_COPYNUM_EST_HOME}/scripts/results/general/combine-est-bwa-outputs.py --use_length_strata k sequence-labels.csv ../../../../../../k$k/aln/${name}_aln-counts.tsv
python ${WGS_COPYNUM_EST_HOME}/scripts/results/general/compute-summary-stats.py aln-est_counts.csv counts summary-stats
cd ../$est_results_folder
python ${WGS_COPYNUM_EST_HOME}/scripts/results/general/combine-est-bwa-outputs.py --use_length_strata k ../../../../../../k$k/results/full/sequence-labels.csv ../../../../../../k$k/aln/${name}_aln-counts.tsv
python ${WGS_COPYNUM_EST_HOME}/scripts/results/general/compute-summary-stats.py aln-est_counts.csv counts summary-stats

set +x

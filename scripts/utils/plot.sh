#!/bin/bash

use_oom_len_gps=""

while getopts "o" opt; do
  case $opt in
    o)
      use_oom_len_gps="--use_oom_len_gps"
      ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      ;;
  esac
done

set -x

# Create plots
mkdir -p plots
mkdir -p plots/observed
python ${WGS_COPYNUM_EST_HOME}/scripts/results/general/component_densities_plot.py $use_oom_len_gps full/seq-est-and-aln.csv summary/length_gp_stats.csv summary/copynumber_params.csv plots/observed component-densities
if [ -z "$(ls -A plots/observed)" ]; then
  rm -rf plots/
fi

set +x

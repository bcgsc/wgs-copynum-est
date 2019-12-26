#!/bin/sh

python ${WGS_COPYNUM_EST_HOME}/scripts/results/general/compute-summary-stats.py aln-est_counts.csv counts summary-stats
for i in `ls aln-est_counts_gte*csv`; do
  e=$(echo $i | perl -ne 'chomp; if (/aln-est_counts_(gte\d+lt(e\d+|inf))\.csv/) { print "$1"; }')
  python ${WGS_COPYNUM_EST_HOME}/scripts/results/general/compute-summary-stats.py aln-est_counts_${e}.csv counts_${e} summary-stats_${e}
done
for i in 100 1000 10000; do
  python ${WGS_COPYNUM_EST_HOME}/scripts/results/general/compute-summary-stats.py aln-est_counts_lt${i}.csv counts_lt${i} summary-stats_lt${i}
done
python ${WGS_COPYNUM_EST_HOME}/scripts/results/general/compute-summary-stats.py aln-est_counts_gte10000.csv counts_gte10000 summary-stats_gte10000


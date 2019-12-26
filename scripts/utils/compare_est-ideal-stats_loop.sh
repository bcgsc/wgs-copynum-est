#!/bin/sh

set -x

python ${WGS_COPYNUM_EST_HOME}/scripts/results/general/compare_est-to-ideal.py summary/summary-stats/summary-stats_full.csv ideal/stats/stats_agg.csv summary/summary-stats/compare_ideal/ agg
for i in `ls summary/summary-stats/summary-stats_gte*full.csv`; do
  e=$(echo $i | perl -ne 'chomp; if (/summary\/summary-stats\/summary-stats_(gte\d+lt(e\d+|inf))_full\.csv/) { print "$1"; }')
  python ${WGS_COPYNUM_EST_HOME}/scripts/results/general/compare_est-to-ideal.py $i ideal/stats/stats_len-${e}.csv summary/summary-stats/compare_ideal/ ${e}
done

set +x

#!/bin/bash

set -x

mkdir -p out
cd hist
${WGS_COPYNUM_EST_HOME}/scripts/genomescope/ntcard-to-jellyfish.sh
for f in `ls`; do
  k=$(echo $f | perl -ne 'chomp; if (/.*k(\d+)\.hist/) { print "$1"; }')
  mkdir -p ../out/k$k
  ${WGS_COPYNUM_EST_HOME}/scripts/genomescope/genomescope.R -i kmer-freq_k${k}.hist -o ../out/k$k -k $k
done

set +x

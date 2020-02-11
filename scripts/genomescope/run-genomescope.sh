#!/bin/bash

first=$1
increment=$2
last=$3
prefix=$4
reads1=$5
reads2=$6
readlen=$7

set -x

mkdir -p hist
cd hist
ntcard -t2 -k21 -p $prefix ../$reads1 ../$reads2
cd ..
${WGS_COPYNUM_EST_HOME}/scripts/explore/run-and-plot_ntcard.sh $first $increment $last $prefix $reads1 $reads2

mkdir out
cd hist
${WGS_COPYNUM_EST_HOME}/scripts/utils/ntcard-to-jellyfish.sh
for f in `ls`; do
  k=$(echo $f | perl -ne 'chomp; if (/.*k(\d+)\.hist/) { print "$1"; }')
  /gsc/software/linux-x86_64-centos6/R-3.3.2/lib64/R/bin/Rscript ${WGS_COPYNUM_EST_HOME}/scripts/genomescope/genomescope.R $f $k $readlen ../out/k$k
done

set +x

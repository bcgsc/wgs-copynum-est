#!/bin/bash

first=$1
increment=$2
last=$3
prefix=$4
reads1=$5
reads2=$6

mkdir hist
cd hist
ntcard -t2 -k$(seq -s ',' $first $increment $last) -p $prefix ../$reads1 ../$reads2
cd ..
mkdir plots
python ${WGS_COPYNUM_EST_HOME}/scripts/explore/plotting/plot_ntcard-output.py hist/ kmer-freq plots

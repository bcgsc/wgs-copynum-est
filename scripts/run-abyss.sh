#!/bin/bash

set -eu -o pipefail

# for OpenMPI
export PATH=/home/benv/bin/mpi:$PATH
export LD_LIBRARY_PATH=/home/benv/lib

# OpenMPI tweaks
eager_limit=16384
export mpirun="mpirun --mca btl_sm_eager_limit $eager_limit --mca btl_openib_eager_limit $eager_limit"

# for ABySS binaries
export PATH=/projects/btl/benv/arch/xhost/abyss-2.0.2/jemalloc-4.5.0/maxk256/bin:$PATH

# input reads
read1=/projects/btl/datasets/celegans/N2strain/DRR008444_1.fastq
read2=/projects/btl/datasets/celegans/N2strain/DRR008444_2.fastq

# k-mer size
k=$1
shift

# create and switch to assembly dir
dir=k$k
mkdir -p $dir
cd $dir

abyss-pe v=-v k=$k name=celegans np=12 j=12 in="$read1 $read2" "$@"

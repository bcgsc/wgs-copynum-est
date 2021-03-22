#!/bin/bash

export TMPDIR=/var/tmp

if [ $# -ne 5 ]; then
	echo "Usage: $(basename $0) <kmer size> <num threads> <PE reads file 1> <PE reads file 2> <output prefix> " >&2
	exit 1
fi

set -eux -o pipefail

# Prevents MPI deadlocks at higher k values
export mpirun="mpirun --mca btl_sm_eager_limit 16000 --mca btl_openib_eager_limit 16000"

k=$1; shift
j=$1; shift
reads1=$1; shift
reads2=$1; shift
prefix=$1; shift

# Choosing the ABySS binary with the closest maxk to the specified k
if [ $k -le 96 ]; then
	  maxk=96
elif [ $k -le 128 ]; then
	  maxk=128
elif [ $k -le 160 ]; then
	  maxk=160
elif [ $k -le 192 ]; then
	  maxk=192
elif [ $k -le 224 ]; then
	  maxk=224
elif [ $k -le 256 ]; then
	  maxk=256
else
	  echo "No ABYSS binary available for k > 256!" >&2
	  exit 1
fi

# ABySS version to use
export PATH=/gsc/btl/linuxbrew/Cellar/abyss/2.2.3/bin/:/gsc/btl/linuxbrew/bin:$PATH
# put `zsh` on PATH so that `abyss-pe` will zsh-profile assembly commands
export PATH=/gsc/btl/linuxbrew/Cellar/zsh/5.4.2_3/bin:$PATH

# run the assembly
abyss_bin=/gsc/btl/linuxbrew/Cellar/abyss/2.2.3/bin
/usr/bin/time -f "pctCPU=%P avgmem=%K maxRSS=%M elapsed=%E cpu.sys=%S cpu.user=%U" ${abyss_bin}/abyss-pe \
    name=$prefix k=$k j=$j np=$j v=-v in="$reads1 $reads2" $prefix-2.dot ABYSS_OPTIONS="-b0"

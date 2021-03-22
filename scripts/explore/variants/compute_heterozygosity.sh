#!/bin/bash

set -x

variants=$1; shift
ref=$1; shift
outfile=$1; shift

het_count=(`wc -l $variants`)
haploid_genomesize=(`wc -m $ref`)

het_rate=`bc -l <<< "$het_count/$haploid_genomesize"`
echo "Figures below are approximate." >> "$outfile"
printf "Number of filtered variant sites: " >> "$outfile"
printf "%s\n" "$het_count" >> "$outfile"
printf "Genome size: " >> "$outfile"
printf "%s\n" "$haploid_genomesize" >> "$outfile"
printf "Per-base heterozygosity rate: " >> "$outfile"
printf "%f\n" "$het_rate" >> "$outfile"

set +x

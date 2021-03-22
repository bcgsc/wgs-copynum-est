#!/bin/bash

set -x

refpath=$1; shift
reads1=$1;shift
reads2=$1;shift
bwamem_k=$1;shift
mpileup_d=$1;shift
mpileup_t=$1;shift
call_t=$1;shift

bwa mem -a -k $bwamem_k $refpath $reads1 $reads2 | samtools view - -bS | samtools sort - -o reads-ref-aln.bam
bcftools mpileup -ABC -d${mpileup_d} -a AD,DP,SCR -o mpileup.vcf --threads=${mpileup_t} -f $refpath reads-ref-aln.bam
bcftools call -mv -Oz --threads=${call_t} -o variants.vcf.gz mpileup.vcf

set +x

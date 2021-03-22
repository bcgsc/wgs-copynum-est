#!/bin/bash

haploid=""
readfiles_prefix="diploid.fastq"
mutation_rate="0.001"
basequal_std="4"

while getopts "e:E:L:C:Q:r:H" opt; do
  case $opt in
    e)
      read1_error=$OPTARG
      ;;
    E)
      read2_error=$OPTARG
      ;;
    L)
      readlen=$OPTARG
      ;;
    C)
      perbase_read_cvg=$OPTARG
      ;;
    H)
      haploid="-H"
      readfiles_prefix="haploid.fastq"
      ;;
    Q)
      basequal_std=$OPTARG
      ;;
    r)
      mutation_rate=$OPTARG
      ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      ;;
  esac
done

shift $((OPTIND-1))
refpath=$1

set -x

# Simulate mutations to reference, and reads
dwgsim -e $read1_error -E $read2_error -r $mutation_rate -C $perbase_read_cvg -1 $readlen -2 $readlen -n $readlen -Q $basequal_std $haploid $refpath $readfiles_prefix

mkdir -p reads
mv *fastq reads/
mkdir -p mutations
mv *mutations.txt mutations/
mv *mutations.vcf mutations/

cd mutations
if [ -n "$haploid" ]; then
  haploid="haploid"
else
  haploid="diploid"
fi
python ${WGS_COPYNUM_EST_HOME}/scripts/utils/simulations/write_mutated_reference.py $haploid ../$refpath ${readfiles_prefix}.mutations.txt

# index mutated reference
filename=$(basename -- "../$refpath")
bwa index ${filename%.*}.mutated.fa

set +x

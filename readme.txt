---------
Workflow (from copy-num-est root directory)
---------

Note: <HOME> indicates project home, e.g. /projects/btl/yflim/copy-num-est

1. Can be done just once per strain/whatever (defined as association with one reference genome)

cd to species data folder, e.g.:
$ cd data/celegans

Create folder for strain, e.g.:
$ mkdir N2
$ cd N2

Create folder for reference, e.g.:
$ mkdir reference

Create folder for datasets, e.g.:
$ mkdir datasets

Create folder(s) for simulations, e.g.:
$ mkdir simulations
$ cd simulations
$ mkdir 50x
$ cd 50x
$ mkdir diploid
$ cd diploid
$ dwgsim -e 0.001-0.015 -E 0.002-0.03 -d 150 -C 50 -1 150 -2 150 -n 100 ../../../reference_genomes/celegans.N2.genome.fa diploid.fastq
$ mkdir haploid
$ cd haploid
$ dwgsim -e 0.001-0.015 -E 0.002-0.03 -d 150 -C 50 -1 150 -2 150 -n 100 -H ../../../reference_genomes/celegans.N2.genome.fa diploid.fastq

2. Can be done just once per dataset (including each set of simulated reads). (Starting from species folder, data/celegans.)

Create folder for dataset, e.g.:
$ mkdir datasets/DRR008444/

$ cd datasets/DRR008444/

Copy dataset read files into folder, e.g.:
$ cp <read1_filepath> .

$ mkdir k80
$ cd k80

Assemble reads, e.g.:
$ <HOME>/scripts/run-abyss.sh <HOME>/data/celegans/N2/datasets/DRR008444/DRR008444_1.fastq <HOME>/data/celegans/N2/datasets/DRR008444/DRR008444_2.fastq 80
ABySS creates a "k-" subfolder, e.g. k80, for its output files:
$ mv k80 abyss-out

Create folder for alignment, e.g.:
$ mkdir aln
$ mkdir aln/w5d50

Run BWA-mem to align unitigs to reference; output to SAM file. E.g.:
$ bwa mem -a -k 80 -w 5 -d 50 -c 10000 ../../../reference/<genome_fa> abyss-out/<unitigs_fa> > aln/w5d50/<ref_aln_sam>
$ cd aln/w5d50/

Edit out header from <ref_aln_sam>
$ cp <ref_aln_sam> <ref_aln_sam_cp>
$ vim <ref_aln_sam_cp>

Parse SAM; write to <ref_aln_counts_csv>
$ python /projects/btl/yflim/copy-num-est/scripts/utils/sam-parse.py <ref_aln_sam_cp> <ref_aln_counts_csv> <error_log> <missing_seqs_log>

Remove error and missing logs if empty
$ rm <error_log>
$ rm <missing_seqs_log>
$ cd ../.. 

Create folder for estimation results:
$ mkdir results
$ mkdir results/aln-est

3. Create folder for specific estimator version results and file describing version details; run estimator, which writes to output files in <output_dir>
$ mkdir results/aln-est/<output_dir>
$ vim results/aln-est/<output_dir>/meta.txt
$ python <HOME>/scripts/est.py abyss-out/<unitigs_fasta> <k> results/aln-est/<output_dir>

Combine estimation and reference alignment data; output goes to <seq_aln_and_est>, and <aln_est_counts> and <aln_est_ranks> files
$ cd results/aln-est/<output_dir>/
$ python <HOME>/scripts/results/process/combine-est-bwa-outputs.py <est_seq_labels> ../../../aln/w5d50/<ref_aln_counts_csv> <number_of_seqs>
Create folders for "full" and summary results; move files accordingly:
$ mkdir full
$ mkdir summary
$ mv <seq_aln_and_est> full
$ mv <est_seq_labels> full
...

4. Compute summary stats; repeat for the various groups (e.g. sequences shorter than 100bps, etc.)
$ cd summary
$ python <HOME>/scripts/results/process/compute-summary-stats.py <aln_est_counts> <aln_est_ranks> <counts> <counts_nb> <summary_stats> <summary_stats_nb> 
 

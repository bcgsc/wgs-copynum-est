---------
Workflow (from copy-num-est root directory)
---------

1. Can be done just once per dataset

cd to species data folder, e.g.:
$ cd data/celegans

Create folder for alignment, e.g.:
$ mkdir assemblies/DRR008444/k80/aln/w5d50

Run BWA-mem to align unitigs to reference; output to SAM file. E.g.:
$ bwa mem -a -k 80 -w 5 -d 50 -c 10000 N2/<genome_fa> assemblies/DRR008444/k80/abyss-out/<unitigs_fa> > assemblies/DRR008444/k80/aln/w5d50/<ref_aln_sam>
$ cd assemblies/DRR008444/k80/aln/w5d50/

Edit out header from <ref_aln_sam>
$ cp <ref_aln_sam> <ref_aln_sam_cp>
$ vim <ref_aln_sam_cp>

Parse SAM; write to <ref_aln_counts_csv>
$ python /projects/btl/yflim/copy-num-est/scripts/results/utils/sam-parse.py <ref_aln_sam_cp> <ref_aln_counts_csv> <error_log> <missing_seqs_log>

Remove error and missing logs if empty
$ rm <error_log>
$ rm <missing_seqs_log>
$ cd /projects/btl/yflim/copy-num-est/

2. Run estimator; it writes to <est_seq_labels> and a file recording estimated component parameters
$ python scripts/est.py data/celegans/assemblies/DRR008444/k80/abyss-out/<unitigs_fasta> <k>

Combine estimation and reference alignment data; output goes to <seq_aln_and_est>, and <aln_est_counts> and <aln_est_ranks> files
$ python scripts/results/process/combine-est-bwa-outputs.py <est_seq_labels> data/celegans/assemblies/DRR008444/k80/aln/k80w5d50/<ref_aln_counts_csv> <number_of_seqs>

3. Compute summary stats; repeat for the various groups (e.g. sequences shorter than 100bps, etc.)
$ python scripts/results/process/compute-summary-stats.py <aln_est_counts> <aln_est_ranks> <summary_stats>
 
Move results, e.g.:
$ mv params.csv data/celegans/assemblies/DRR008444/k80/results/aln-est/20180329/full/
$ mv <est_seq_labels> data/celegans/assemblies/DRR008444/k80/results/aln-est/20180329/full/
$ mv <seq_aln_and_est> data/celegans/assemblies/DRR008444/k80/results/aln-est/20180329/full/
$ mv <aln_est_counts> data/celegans/assemblies/DRR008444/k80/results/aln-est/20180329/summary/
$ mv <aln_est_ranks> data/celegans/assemblies/DRR008444/k80/results/aln-est/20180329/summary/
$ mv <summary_stats> data/celegans/assemblies/DRR008444/k80/results/aln-est/20180329/summary/

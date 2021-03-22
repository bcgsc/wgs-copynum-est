Example workflow (from project root directory)

Note:
a. <HOME> indicates project home, e.g. /projects/btl/yflim/copy-num-est
b. Dataset (genome) used as example(s) below not consistent; e.g. some use C. elegans, while others use Bovine adenovirus, or some use real and some simulated data.
c. Relative locations of folders in examples below may not always be accurate.


1. Can be done just once per strain/whatever (defined as association with one reference genome)

cd to species data folder, e.g.:
$ cd data/celegans

Create folder for strain, e.g.:
$ mkdir N2
$ cd N2

Create folder for reference, e.g.:
$ mkdir reference
...
If necessary, concatenate (chromosome-level) FASTA files into a single consolidated FASTA file with 2 lines (name & sequence) per chromosome (first remove irrelevant FASTA files, including compressed versions):
$ <HOME>/scripts/utils/concat_chromosomes.py <shared_prefix>
E.g.
$ <HOME>/scripts/utils/concat_chromosomes.py Escherichia_coli
Index for BWA:
$ bwa index reference/celegans.N2.genome.fa

Create folder for datasets, e.g.:
$ mkdir datasets

Create folder(s) for simulations, e.g.:
$ mkdir simulations
$ cd simulations
$ mkdir 50x
$ cd 50x
$ <HOME>/scripts/utils/simulations/sim_mutate-ref_idx.sh -e 0.001-0.005 -E 0.002-0.01 [-r <mutation_rate>] -L 150 -C 30 [-H] ../../../reference_genomes/celegans.N2.genome.fa

2. Can be done just once per dataset (including each set of simulated reads). (Starting from strain folder, data/celegans/N2.)

Create folder for (real genome) dataset, e.g. (first cd back to datasets folder):
$ cd ../..
$ mkdir DRR008444/

$ cd DRR008444/

Create folder for reads and copy dataset read files into it, e.g.:
$ mkdir reads
$ cp <reads_filepath> reads

Run ntcard over a range of possible ks, in order to choose a few best candidates based on output k-mer frequency (read coverage) histograms:
$ cd ..
$ mkdir ntcard
$ cd ntcard
$ <HOME>/scripts/explore/run-and-plot_ntcard.sh 60 5 120 kmer-freq ../reads/diploid.fastq.bwa.read1.fastq ../reads/diploid.fastq.bwa.read2.fastq

Assemble reads, e.g.:
$ cd ..
$ mkdir k70
$ cd k70
<bwa_k>: minimum seed length, i.e. matches shorter than this will be missed. Use case: real genomic datasets. Should be equal to k for simulated datasets.
$ <HOME>/scripts/utils/abyss-and-aln.sh 70 2 ../reads/haploid.fastq.bwa.read1.fastq ../reads/haploid.fastq.bwa.read2.fastq bovine-adenovirus <bwa_k> 1 ../mutations/Bovine_adenovirus_6.mutated.fa
$ cd ..
$ mkdir k75
...

$ cd k<k>
$ <HOME>/scripts/utils/est-and-eval.sh [-H|h|e|o|l|i] [-c longest_seqs_peak_expected_cpnum] [-t seq_identity_threshold] celegans <k>
$ cd results
$ <HOME>/scripts/utils/plot.sh [-o][i]

3. (Can be done earlier if omitting estimation results.) Plot alignment feature (e.g. edit distance, perfect match) distributions by sequence length and depth, and optionally by estimated copy number.
$ python <HOME>/scripts/explore/plotting/alignment-stats_plots.py <k> <ref_aln_counts_csv> abyss-out/<unitigs_fasta> <alnmt_stats_plots_folder> [aln_est_counts]

4. Fit, evaluate, and compare with GenomeScope (requires running run-and-plot_ntcard.sh [see above] first)
$ mkdir ntcard/genomescope
$ cp -r ntcard/hist ntcard/genomescope
$ cd ntcard/genomescope
[$ <HOME>/scripts/genomescope/run-genomescope.sh
$ cd ../..
$ <HOME>/scripts/genomescope/eval-genomescope.sh 80 celegans 2|3 ntcard/hist/kmer-freq ntcard/genomescope/out/converged/k80/ gs-cpnum-results est-cpnum-results

5. Unicycler comparison
From species-, depth-, k-specific folder, e.g. <HOME>/data/ecoli/datasets/simulations/30x/k60
$ mkdir unicycler
Note WTF: "Error: --kmers must be comma-separated odd integers without spaces (example: --kmers 21,31,41)". So for odd k:
$ unicycler -1 ../reads/haploid.fastq.bwa.read1.fastq -2 ../reads/haploid.fastq.bwa.read2.fastq -o unicycler/ --verbosity 2 --min_fasta_length <k> --keep 3 --spades_path /gsc/btl/linuxbrew/bin/spades.py [--no_correct] --kmers <k> --depth_filter 0.1
$ cd unicycler
$ less unicycler.log
$ nohup <HOME>/scripts/unicycler/compare.sh <k> <last_multiplicity_line> <# of multiplicity lines> eval 1 ../../mutated/Escherichia_coli.mutated.fa &

6. Real genome
$ cd <dataset_name>
$ <HOME>/scripts/explore/variants/reads-to-vcf.sh <ref_path> <reads1> <reads2> <bwamem_k> <mpileup_d> <mpileup_t> <call_t>
$ mkdir aln
$ mv reads-ref-aln.bam mpileup.vcf reads-to-vcf.log variants.vcf.gz aln/
$ cd aln/
$ python <HOME>/scripts/explore/variants/vcf_parse.py variants.vcf.gz variants_parsed.tsv
$ cd k<k>
$ <HOME>/scripts/utils/results/est-and-eval.sh -[H|h|e|o|l|i] [-c longest_seqs_peak_expected_cpnum] [-t <seq_iden_threshold>] celegans <k>
$ <HOME>/scripts/real-genomes/sam_parse.py aln/${name}_aln.sam aln/celegans_aln_parsed.tsv

7. Simulating and validating high-heterozygosity data:
$ cd <HOME>/data/celegans/N2/datasets/simulations/high-heterozygosity
$ mkdir high-heterozygosity
$ cd high-heterozygosity
$ mkdir 30X
$ <HOME>/scripts/utils/simulations/sim_mutate-ref_idx.sh -e 0.001-0.005 -E 0.002-0.01 [-r <mutation_rate>] -L 150 -C 30 [-H] <refpath>
$ <HOME>/scripts/explore/variants/reads-to-vcf.sh <refpath> reads/<reads1> reads/<reads2> <bwamem_k> <mpileup_d> <mpileup_t> <call_t>
$ mkdir aln
$ mv reads-ref-aln.bam mpileup.vcf reads-to-vcf.log variants.vcf.gz aln/
$ cd aln
$ python <HOME>/scripts/explore/variants/vcf_parse.py variants.vcf.gz variants_parsed.tsv
$ python <HOME>/scripts/explore/variants/filter_variants.py --frac_mapq0 0 --alt_depth 10 --alt_depth_ratio 0.3 --map_qual 60 --alt_call_qual 65 --heterogygous variants_parsed.tsv variants_parsed_filtered.tsv
$ <HOME>/scripts/explore/variants/compute_heterozygosity.sh variants_parsed_filtered.tsv <refpath> heterozygosity.txt

8. Miscellaneous:
$ cd k70
$ python <HOME>/scripts/utils/fa_to_csv.py abyss-out/celegans-2.fa abyss-out/celegans-2.csv

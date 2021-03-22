import argparse
import pandas as pd

argparser = argparse.ArgumentParser(description="Filter parsed variant calls output by bcftools mpileup based on variant type, read depth, etc.")
argparser.add_argument('--alt_call_qual', type=float, nargs='?', default=0, help='Minimum variant call quality (phred-scaled probability of being incorrect)')
argparser.add_argument('--alt_depth', type=float, nargs='?', default=0, help='Minimum alt allele read depth')
argparser.add_argument('--alt_depth_ratio', type=float, nargs='?', default=0, help='Minimum ratio of alt allele to total read depth')
argparser.add_argument('--map_qual', type=float, nargs='?', default=0, help='Minimum average mapping quality')
argparser.add_argument('--frac_mapq0', type=float, nargs='?', default=1, help='Maximum fraction of reads with mapping quality 0')
argparser.add_argument('--heterogygous', action='store_true', help='Select only heterozygous variants')
argparser.add_argument("variants_file", type=str, help="Path of file listing variants")
argparser.add_argument("output_file", type=str, help="Output file path")
args = argparser.parse_args()

variants = pd.read_csv(args.variants_file, delimiter='\t')
variants['alt_depth_ratio'] = variants.alt_allele_depth / variants.avg_read_depth
filtered = variants.loc[variants.alt_call_qual >= args.alt_call_qual]
filtered = filtered.loc[filtered.alt_allele_depth >= args.alt_depth]
filtered = filtered.loc[filtered.alt_depth_ratio >= args.alt_depth_ratio]
filtered = filtered.loc[filtered.avg_map_qual >= args.map_qual]
filtered = filtered.loc[filtered.fraction_mapq_0 <= args.frac_mapq0]
if args.heterogygous:
    filtered = filtered.loc[filtered.monoallelic_variant == 0]

filtered.to_csv(args.output_file, index=False, sep='\t')

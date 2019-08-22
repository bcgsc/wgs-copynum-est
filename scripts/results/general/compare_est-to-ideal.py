import argparse
import numpy as np
import os
import pandas as pd

argparser = argparse.ArgumentParser(description='Measure copy number estimator performance relative to estimated best-possible classifier performance')
argparser.add_argument('est_stats', type=str, help='CSV table representing summary statistics for some sequence group: 1 row per copy number, and columns TPR, [FNR,] PPV, [FDR,] F1')
argparser.add_argument('ideal_stats', type=str, help='CSV table representing estimated ideal summary statistics for some sequence group: 1 row per copy number, and columns TPR, [FNR,] PPV, [FDR,] F1')
argparser.add_argument('output_dir', type=str, help='Directory to which output file should be written')
argparser.add_argument('output_filename', type=str, help='Output file prefix')
args = argparser.parse_args()

ideal_stats = pd.read_csv(args.ideal_stats)
ideal_stats.rename(inplace = True, columns = { 'Copy #': 'copynum' })
ideal_stats.set_index('copynum', inplace=True)
est_stats = pd.read_csv(args.est_stats)
est_stats.rename(inplace = True, columns = { 'Copy #': 'copynum' })
est_stats.set_index('copynum', inplace=True)
est_stats = est_stats.reindex(ideal_stats.index)

ratios = pd.DataFrame(np.nan, index = est_stats.index, columns = ['TPR', 'PPV', 'F1'])
ratios['TPR'] = est_stats.TPR / ideal_stats.TPR
ratios['PPV'] = est_stats.PPV / ideal_stats.PPV
ratios['F1'] = est_stats.F1 / ideal_stats.F1
ratios.to_csv(os.path.join(args.output_dir, args.output_filename + '.csv'), index_label = 'Copy #')

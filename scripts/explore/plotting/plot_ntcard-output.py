import argparse
import glob
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
from scipy import stats
import seaborn as sns

argparser = argparse.ArgumentParser(description='Plot ntCard output histograms within current folder')
argparser.add_argument('hist_folder', type=str, help='Folder containing ntCard output histograms')
argparser.add_argument('prefix', type=str, help='Prefix for all ntCard output files within current folder')
argparser.add_argument('plots_folder', type=str, help='Folder name for output histogram plot files')
args = argparser.parse_args()


for f in glob.glob(args.hist_folder + '/' + args.prefix + '_k*.hist'):
  print('plotting output (histogram) in ' + f)
  ntcard_stats = pd.read_csv(f, sep='\t', header=None)
  hist = ntcard_stats.iloc[2:]
  hist.rename(columns = { 0: 'cvg', 1: 'kmers' }, inplace=True)
  hist['cvg'] = hist.cvg.astype(int)
  cvg_long = np.repeat(hist.cvg.values, hist.kmers.values)
  hist_full, singleton_label = np.histogram(cvg_long, bins=range(1, hist.cvg.max() + 2)), False
  if hist_full[0][0] > 3 * np.max(hist_full[0][1:]):
    singleton_label = True
  cvg_long_truncated = cvg_long[cvg_long <= np.percentile(cvg_long, 100 - stats.variation(cvg_long) * 0.5)]
  bins = np.unique(cvg_long_truncated)[int(singleton_label):].tolist() + [cvg_long_truncated[-1] + 1]
  ax = sns.distplot(cvg_long_truncated, bins = bins, kde = False)
  k = re.search(args.prefix + '_k(\d{2,3}).hist$', f).group(1)
  ax.set(xlabel = 'k-mer read coverage' + singleton_label * (' (unique k-mers: ' + str(hist_full[0][0]) + ')'),
      ylabel = '# distinct k-mers', xticks = bins, title = 'k = ' + k)
  ax.get_figure().savefig(args.plots_folder + '/' + args.prefix + '_k' + k + '.png')
  plt.clf()


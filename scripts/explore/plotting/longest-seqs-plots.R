#!/usr/bin/env Rscript

library(ggplot2)

data = read.csv("gen-workspace/celegans-1-readdata.csv")
length = data[,2]
cvgs_sum = data[,3]
k = 80
seq_kmer_count = length - k + 1
avg_kmer_cvg = cvgs_sum / seq_kmer_count
len_cvg = data.frame(len = length, cvg = avg_kmer_cvg)

sorted_length = sort(length)
len_quantiles = quantile(sorted_length, seq(0, 1, 0.02))
len_quantiles_factor = factor(len_quantiles, levels=unique(len_quantiles), ordered=TRUE)
lengths_cut = cut(length, as.numeric(levels(len_quantiles_factor)), include.lowest = TRUE)
len_cvg['lengths_cut'] = lengths_cut

uniq_len_ranges = sort(unique(lengths_cut))
len_ranges_count = length(uniq_len_ranges)
longest = len_cvg[len_cvg$lengths_cut == uniq_len_ranges[len_ranges_count],]



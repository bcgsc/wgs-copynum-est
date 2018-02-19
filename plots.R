#!/usr/bin/env Rscript

library(ggplot2)

data = read.csv("celegans-1-readdata.csv")
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

# Histograms
# produces phantom plots at the end, but whatever
xlab = "average k-mer depth"
len_cvg['lengths_cut'] = lengths_cut
pdf(file="~/temp/copy-num-est/histograms/histograms-with-densities-lengthwise.pdf")
uniq_len_ranges = sort(unique(lengths_cut))
graphs = vector("list", 3 * length(uniq_len_ranges))
lapply(uniq_len_ranges, function(i) {
  curr_title = paste("sequence lengths in ", i)
  currlen_seqs = len_cvg[len_cvg$lengths_cut == i,]
  j = 3 * (as.numeric(i) - 1)
  curr = currlen_seqs[currlen_seqs$cvg <= 400,]
  if (dim(curr)[1] > 0) {
  graphs[[j+1]] = ggplot(curr, aes(curr$cvg)) + labs(x = xlab, title = curr_title) + geom_histogram(binwidth=1, alpha=0.2, aes(y = ..density..)) + geom_density(bw=1.5)
  print(graphs[[j+1]])
  }
  curr = currlen_seqs[(currlen_seqs$cvg > 400) & (currlen_seqs$cvg <= 2000),]
  if (dim(curr)[1] > 0) {
  graphs[[j+2]] = ggplot(curr, aes(curr$cvg)) + labs(x = xlab, title = curr_title) + geom_histogram(binwidth=1, alpha=0.2, aes(y = ..density..)) + geom_density(bw=1.5) + xlim(400,NA)
  print(graphs[[j+2]])
  }
  curr = currlen_seqs[currlen_seqs$cvg > 2000,]
  if (dim(curr)[1] > 0) {
  graphs[[j+3]] = ggplot(curr, aes(curr$cvg)) + labs(x = xlab, title = curr_title) + geom_histogram(binwidth=1, alpha=0.2, aes(y = ..density..)) + geom_density(bw=1.5) + xlim(2000, NA)
  print(graphs[[j+3]])
  }
})
invisible(lapply(graphs, print))
dev.off()

# Other
xlab = "sequence length"
ylab = "average k-mer depth"

# Boxplots
ylim = boxplot.stats(len_cvg$cvg)$stats[5]
lens_lte109 = len_cvg[len_cvg$len <= 109,]
boxplot0_lens_lte109 = ggplot(lens_lte109, aes(lens_lte109$lengths_cut, lens_lte109$cvg)) + labs(x = xlab, y = ylab) + geom_boxplot()
boxplot1_lens_lte109 = boxplot0_lens_lte109 + coord_cartesian(ylim = c(0, ylim*3))
lens_gt109 = len_cvg[len_cvg$len > 109,]
boxplot0_lens_gt109 = ggplot(lens_gt109, aes(lens_gt109$lengths_cut, lens_gt109$cvg)) + labs(x = xlab, y = ylab) + geom_boxplot()
boxplot1_lens_gt109 = boxplot0_lens_gt109 + coord_cartesian(ylim = c(0, ylim))
pdf(file="~/temp/copy-num-est/boxplots.pdf", width=11, height=8, paper="USr")
invisible(lapply(list(boxplot0_lens_lte109, boxplot1_lens_lte109, boxplot0_lens_gt109, boxplot1_lens_gt109), print))
dev.off()

# scatterplots
scatter_all_0 = ggplot(len_cvg, aes(length, avg_kmer_cvg)) + labs(x = xlab, y = ylab) + geom_point(alpha = 0.1)
scatter_all_1 = scatter_all_0 + coord_cartesian(ylim = c(0, 30))

lens_lte25000 = len_cvg[len_cvg$len <= 25000,]
scatter_lte25000_0 = ggplot(lens_lte25000, aes(lens_lte25000$len, lens_lte25000$cvg)) + labs(x = xlab, y = ylab) + geom_point(alpha = 0.1)
scatter_lte25000_1 = scatter_lte25000_0 + coord_cartesian(ylim = c(0, 30))

lens_gt25000 = len_cvg[len_cvg$lengths > 25000,]
scatter_gt25000_0 = ggplot(lens_gt25000, aes(lens_gt25000$lengths, lens_gt25000$cvg)) + labs(x = xlab, y = ylab) + geom_point(alpha = 0.1)
scatter_gt25000_1 = scatter_gt25000_0 + coord_cartesian(ylim = c(0, 30))

lens_lte200 = len_cvg[len_cvg$len <= 200,]
scatter_lte200_0 = ggplot(lens_lte200, aes(lens_lte200$len, lens_lte200$cvg)) + labs(x = xlab, y = ylab) + geom_point(alpha = 0.1)
scatter_lte200_1 = scatter_lte200_0 + coord_cartesian(ylim = c(0, 30))

lens_lte300 = len_cvg[len_cvg$len <= 300,]
scatter_lte300_0 = ggplot(lens_lte300, aes(lens_lte300$len, lens_lte300$cvg)) + labs(x = xlab, y = ylab) + geom_point(alpha = 0.1)
scatter_lte300_1 = scatter_lte300_0 + coord_cartesian(ylim = c(0, 30))

lens_gt300_lte1000 = len_cvg[(len_cvg$len > 300) & (len_cvg$len <= 1000),]
scatter_gt300_lte1000_0 = ggplot(lens_gt300_lte1000, aes(lens_gt300_lte1000$len, lens_gt300_lte1000$cvg)) + labs(x = xlab, y = ylab) + geom_point(alpha = 0.1)
scatter_gt300_lte1000_1 = scatter_gt300_lte1000_0 + coord_cartesian(ylim = c(0, 100))

lens_gt1000_lte10000 = len_cvg[(len_cvg$len > 1000) & (len_cvg$len <= 10000),]
scatter_gt1000_lte10000_0 = ggplot(lens_gt1000_lte10000, aes(lens_gt1000_lte10000$len, lens_gt1000_lte10000$cvg)) + labs(x = xlab, y = ylab) + geom_point(alpha = 0.1)
scatter_gt1000_lte10000_1 = scatter_gt1000_lte10000_0 + coord_cartesian(ylim = c(0, 30))

lens_gt10000 = len_cvg[len_cvg$len > 10000,]
scatter_gt10000_0 = ggplot(lens_gt10000, aes(lens_gt10000$len, lens_gt10000$cvg)) + labs(x = xlab, y = ylab) + geom_point(alpha = 0.1)
scatter_gt10000_1 = scatter_gt10000_0 + coord_cartesian(ylim = c(0, 30))

lens_gt10000_lte50000 = len_cvg[(len_cvg$len > 10000) & (len_cvg$len <= 50000),]
scatter_gt10000_lte50000_0 = ggplot(lens_gt10000_lte50000, aes(lens_gt10000_lte50000$len, lens_gt10000_lte50000$cvg)) + labs(x = xlab, y = ylab) + geom_point(alpha = 0.1)
scatter_gt10000_lte50000_1 = scatter_gt10000_lte50000_0 + coord_cartesian(ylim = c(0, 30))

pdf(file="~/temp/copy-num-est/scatterplots.pdf", width=11, height=8, paper="USr")
scatter = list(scatter_all_0, scatter_all_1, scatter_lte25000_0, scatter_lte25000_1, scatter_gt25000_0, scatter_gt25000_1, scatter_lte200_0, scatter_lte200_1, scatter_lte300_0, scatter_lte300_1, scatter_gt300_lte1000_0, scatter_gt300_lte1000_1, scatter_gt1000_lte10000_0, scatter_gt1000_lte10000_1, scatter_gt10000_0, scatter_gt10000_1, scatter_gt10000_lte50000_0, scatter_gt10000_lte50000_1)
invisible(lapply(scatter, print))
dev.off()

# scratch

> boxplot0_lens_gt109 = ggplot(lens_gt109, aes(lens_gt109$lengths_cut, lens_gt109$cvg)) + geom_boxplot()
> boxplot0_lens_gt109 = ggplot(lens_gt109, aes(lens_gt109$lengths_cut, lens_gt109$cvg)) + geom_boxplot()
> boxplot1_lens_gt109 = boxplot0_lens_gt109 + coord_cartesian(ylim = c(0, ylim1[2]*3))
> boxplot1_lens_gt109
> boxplot1_lens_gt109 = boxplot0_lens_gt109 + coord_cartesian(ylim = c(0, ylim1[2]*1.5))
> boxplot1_lens_gt109
> boxplot1_lens_gt109 = boxplot0_lens_gt109 + coord_cartesian(ylim = c(0, ylim1[2]*1.2))
> boxplot1_lens_gt109
> boxplot1_lens_gt109 = boxplot0_lens_gt109 + coord_cartesian(ylim = c(0, ylim1[2]))
> boxplot1_lens_gt109
> pdf(file="~/temp/copy-num-est/boxplots.pdf")
> boxplot1_lens_lte109 = boxplot0_lens_lte109 + coord_cartesian(ylim = c(0, ylim1[2]*3))
> lens_lte109 = len_cvg[len_cvg$lengths <= 109,]
> boxplot0_lens_lte109 = 
> boxplot0_lens_lte109 = ggplot(lens_lte109, aes(lens_lte109$lengths_cut, lens_lte109$cvg)) + geom_boxplot()

pdf(file="~/temp/copy-num-est/boxplots.pdf")
xlab = "Sequence length"

ggplot(len_cvg, aes(lengths_cut, cvg)) + geom_boxplot()

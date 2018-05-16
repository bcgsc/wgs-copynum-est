#!/usr/bin/env Rscript

library(ggplot2)

data = read.csv("seq-est-and-aln-oom_20180301.csv")
data['Est..1st.label.factor'] = lapply(data['Est..1st.label'], as.integer)
data['Est..1st.label.factor'] = ifelse(data$Est..1st.label.factor > 4, 5, data$Est..1st.label.factor)
data['Est..1st.label.factor'] = lapply(data['Est..1st.label.factor'], as.factor)

data['Ref..alns.factor'] = lapply(data['Ref..alns'], as.integer)
data['Ref..alns.factor'] = ifelse(data$Ref..alns.factor > 4, 5, data$Ref..alns.factor)
data['Ref..alns.factor'] = lapply(data['Ref..alns.factor'], as.factor)

data_cvglt200 = data[data$Average.depth <= 200,]
data_cvglt400 = data[data$Average.depth <= 400,]
data_cvglt200_1 = data_cvglt200[data_cvglt200$Length < 100,]
data_cvglt400_1 = data_cvglt400[data_cvglt400$Length < 100,]
data_cvglt200_2 = data_cvglt200[(data_cvglt200$Length > 99) & (data_cvglt200$Length < 1000),]
data_cvglt400_2 = data_cvglt400[(data_cvglt400$Length > 99) & (data_cvglt400$Length < 1000),]
data_cvglt200_3 = data_cvglt200[data_cvglt200$Length > 999,]
data_cvglt400_3 = data_cvglt400[data_cvglt400$Length > 999,]

pdf(file="component-density-plots.pdf", width=15)
graphs = vector("list", 8)
graphs[[1]] = ggplot(data_cvglt200, aes(data_cvglt200$Average.depth)) + labs(x = 'avg depth', title = 'Est. component & observed densities', colour='Est. copy # group') + geom_density() + geom_density(aes(colour=data_cvglt200$Est..1st.label.factor))
graphs[[2]] = ggplot(data_cvglt400, aes(data_cvglt400$Average.depth)) + labs(x = 'avg depth', title = 'Est. component & observed densities', colour='Est. copy # group') + geom_density() + geom_density(aes(colour=data_cvglt400$Est..1st.label.factor))
graphs[[3]] = ggplot(data_cvglt200_1, aes(data_cvglt200_1$Average.depth)) + labs(x = 'avg depth', title = 'Est. component & observed densities: sequence lengths < 100', colour='Est. copy # group') + geom_density() + geom_density(aes(colour=data_cvglt200_1$Est..1st.label.factor))
graphs[[4]] = ggplot(data_cvglt400_1, aes(data_cvglt400_1$Average.depth)) + labs(x = 'avg depth', title = 'Est. component & observed densities: sequence lengths < 100', colour='Est. copy # group') + geom_density() + geom_density(aes(colour=data_cvglt400_1$Est..1st.label.factor))
graphs[[5]] = ggplot(data_cvglt200_2, aes(data_cvglt200_2$Average.depth)) + labs(x = 'avg depth', title = 'Est. component & observed densities: sequence lengths 100 to 999', colour='Est. copy # group') + geom_density() + geom_density(aes(colour=data_cvglt200_2$Est..1st.label.factor))
graphs[[6]] = ggplot(data_cvglt400_2, aes(data_cvglt400_2$Average.depth)) + labs(x = 'avg depth', title = 'Est. component & observed densities: sequence lengths 100 to 999', colour='Est. copy # group') + geom_density() + geom_density(aes(colour=data_cvglt400_2$Est..1st.label.factor))
graphs[[7]] = ggplot(data_cvglt200_3, aes(data_cvglt200_3$Average.depth)) + labs(x = 'avg depth', title = 'Est. component & observed densities: sequence lengths >= 1000', colour='Est. copy # group') + geom_density() + geom_density(aes(colour=data_cvglt200_3$Est..1st.label.factor))
graphs[[8]] = ggplot(data_cvglt400_3, aes(data_cvglt400_3$Average.depth)) + labs(x = 'avg depth', title = 'Est. component & observed densities: sequence lengths >= 1000', colour='Est. copy # group') + geom_density() + geom_density(aes(colour=data_cvglt400_3$Est..1st.label.factor))
invisible(lapply(graphs, print))
dev.off()

pdf(file="component-histograms.pdf", width=15)
graphs[[1]] = ggplot(data_cvglt200, aes(Average.depth, fill=Est..1st.label.factor)) + labs(x = 'avg. depth', title = 'Histogram: avg. k-mer depth by est. copy #', fill='Est. copy # group') + geom_histogram(binwidth=1)
graphs[[2]] = ggplot(data_cvglt400, aes(Average.depth, fill=Est..1st.label.factor)) + labs(x = 'avg. depth', title = 'Histogram: avg. k-mer depth by est. copy #', fill='Est. copy # group') + geom_histogram(binwidth=1)
graphs[[3]] = ggplot(data_cvglt200_1, aes(Average.depth, fill=Est..1st.label.factor)) + labs(x = 'avg. depth', title = 'Histogram: avg. k-mer depth by est. copy #, seq. lengths < 100', fill='Est. copy # group') + geom_histogram(binwidth=1)
graphs[[4]] = ggplot(data_cvglt400_1, aes(Average.depth, fill=Est..1st.label.factor)) + labs(x = 'avg. depth', title = 'Histogram: avg. k-mer depth by est. copy #, seq. lengths < 100', fill='Est. copy # group') + geom_histogram(binwidth=1)
graphs[[5]] = ggplot(data_cvglt200_2, aes(Average.depth, fill=Est..1st.label.factor)) + labs(x = 'avg. depth', title = 'Histogram: avg. k-mer depth by est. copy #, seq. lengths 100 to 999', fill='Est. copy # group') + geom_histogram(binwidth=1)
graphs[[6]] = ggplot(data_cvglt400_2, aes(Average.depth, fill=Est..1st.label.factor)) + labs(x = 'avg. depth', title = 'Histogram: avg. k-mer depth by est. copy #, seq. lengths 100 to 999', fill='Est. copy # group') + geom_histogram(binwidth=1)
graphs[[7]] = ggplot(data_cvglt200_3, aes(Average.depth, fill=Est..1st.label.factor)) + labs(x = 'avg. depth', title = 'Histogram: avg. k-mer depth by est. copy #, seq. lengths >= 1000', fill='Est. copy # group') + geom_histogram(binwidth=1)
graphs[[8]] = ggplot(data_cvglt400_3, aes(Average.depth, fill=Est..1st.label.factor)) + labs(x = 'avg. depth', title = 'Histogram: avg. k-mer depth by est. copy #, seq. lengths >= 1000', fill='Est. copy # group') + geom_histogram(binwidth=1)
invisible(lapply(graphs, print))
dev.off()

pdf(file="component-ref-aln-density-plots.pdf", width=15)
graphs = vector("list", 8)
graphs[[1]] = ggplot(data_cvglt200, aes(data_cvglt200$Average.depth)) + labs(x = 'avg depth', title = 'Est. component & observed densities', colour='Copy # group') + geom_density() + geom_density(aes(colour=data_cvglt200$Est..1st.label.factor)) + geom_density(aes(colour=data_cvglt200$Ref..alns.factor), linetype='dashed')
graphs[[2]] = ggplot(data_cvglt400, aes(data_cvglt400$Average.depth)) + labs(x = 'avg depth', title = 'Est. component & observed densities', colour='Copy # group') + geom_density() + geom_density(aes(colour=data_cvglt400$Est..1st.label.factor)) + geom_density(aes(colour=data_cvglt400$Ref..alns.factor), linetype='dashed')
graphs[[3]] = ggplot(data_cvglt200_1, aes(data_cvglt200_1$Average.depth)) + labs(x = 'avg depth', title = 'Est. component & observed densities: sequence lengths < 100', colour='Copy # group') + geom_density() + geom_density(aes(colour=data_cvglt200_1$Est..1st.label.factor)) + geom_density(aes(colour=data_cvglt200_1$Ref..alns.factor), linetype='dashed')
graphs[[4]] = ggplot(data_cvglt400_1, aes(data_cvglt400_1$Average.depth)) + labs(x = 'avg depth', title = 'Est. component & observed densities: sequence lengths < 100', colour='Copy # group') + geom_density() + geom_density(aes(colour=data_cvglt400_1$Est..1st.label.factor)) + geom_density(aes(colour=data_cvglt400_1$Ref..alns.factor), linetype='dashed')
graphs[[5]] = ggplot(data_cvglt200_2, aes(data_cvglt200_2$Average.depth)) + labs(x = 'avg depth', title = 'Est. component & observed densities: sequence lengths 100 to 999', colour='Copy # group') + geom_density() + geom_density(aes(colour=data_cvglt200_2$Est..1st.label.factor)) + geom_density(aes(colour=data_cvglt200_2$Ref..alns.factor), linetype='dashed')
graphs[[6]] = ggplot(data_cvglt400_2, aes(data_cvglt400_2$Average.depth)) + labs(x = 'avg depth', title = 'Est. component & observed densities: sequence lengths 100 to 999', colour='Copy # group') + geom_density() + geom_density(aes(colour=data_cvglt400_2$Est..1st.label.factor)) + geom_density(aes(colour=data_cvglt400_2$Ref..alns.factor), linetype='dashed')
graphs[[7]] = ggplot(data_cvglt200_3, aes(data_cvglt200_3$Average.depth)) + labs(x = 'avg depth', title = 'Est. component & observed densities: sequence lengths >= 1000', colour='Copy # group') + geom_density() + geom_density(aes(colour=data_cvglt200_3$Est..1st.label.factor)) + geom_density(aes(colour=data_cvglt200_3$Ref..alns.factor), linetype='dashed')
graphs[[8]] = ggplot(data_cvglt400_3, aes(data_cvglt400_3$Average.depth)) + labs(x = 'avg depth', title = 'Est. component & observed densities: sequence lengths >= 1000', colour='Copy # group') + geom_density() + geom_density(aes(colour=data_cvglt400_3$Est..1st.label.factor)) + geom_density(aes(colour=data_cvglt400_3$Ref..alns.factor), linetype='dashed')
invisible(lapply(graphs, print))
dev.off()

pdf(file="ref-aln-histograms.pdf", width=15)
graphs = vector("list", 8)
graphs[[1]] = ggplot(data_cvglt200, aes(Average.depth, fill=Ref..alns.factor)) + labs(x = 'avg. depth', title = 'Histogram: avg. k-mer depth by ref. alns.', fill='Copy # group') + geom_histogram(binwidth=1)
graphs[[2]] = ggplot(data_cvglt400, aes(Average.depth, fill=Ref..alns.factor)) + labs(x = 'avg. depth', title = 'Histogram: avg. k-mer depth by ref. alns.', fill='Copy # group') + geom_histogram(binwidth=1)
graphs[[3]] = ggplot(data_cvglt200_1, aes(Average.depth, fill=Ref..alns.factor)) + labs(x = 'avg. depth', title = 'Histogram: avg. k-mer depth by ref. alns., seq. lengths < 100', fill='Copy # group') + geom_histogram(binwidth=1)
graphs[[4]] = ggplot(data_cvglt400_1, aes(Average.depth, fill=Ref..alns.factor)) + labs(x = 'avg. depth', title = 'Histogram: avg. k-mer depth by ref. alns., seq. lengths < 100', fill='Copy # group') + geom_histogram(binwidth=1)
graphs[[5]] = ggplot(data_cvglt200_2, aes(Average.depth, fill=Ref..alns.factor)) + labs(x = 'avg. depth', title = 'Histogram: avg. k-mer depth by ref. alns., seq. lengths 100 to 999', fill='Copy # group') + geom_histogram(binwidth=1)
graphs[[6]] = ggplot(data_cvglt400_2, aes(Average.depth, fill=Ref..alns.factor)) + labs(x = 'avg. depth', title = 'Histogram: avg. k-mer depth by ref. alns., seq. lengths 100 to 999', fill='Copy # group') + geom_histogram(binwidth=1)
graphs[[7]] = ggplot(data_cvglt200_3, aes(Average.depth, fill=Ref..alns.factor)) + labs(x = 'avg. depth', title = 'Histogram: avg. k-mer depth by ref. alns., seq. lengths >= 1000', fill='Copy # group') + geom_histogram(binwidth=1)
graphs[[8]] = ggplot(data_cvglt400_3, aes(Average.depth, fill=Ref..alns.factor)) + labs(x = 'avg. depth', title = 'Histogram: avg. k-mer depth by ref. alns., seq. lengths >= 1000', fill='Copy # group') + geom_histogram(binwidth=1)
invisible(lapply(graphs, print))
dev.off()

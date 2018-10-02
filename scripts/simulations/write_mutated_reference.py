import re
import sys

def finalise_chromosome(c1_seqs, c2_seqs, c1_last, c2_last, full_label, out):
    c1_seqs.append(c1_last)
    c2_seqs.append(c2_last)
    out.write(full_label)
    out.write(''.join(c1_seqs))
    out.write(full_label)
    out.write(''.join(c2_seqs))

def get_base_from_iupac(fr, to):
    if to == 'R':
        if fr == 'A':
            return 'G'
        else:
            return 'A'
    if to == 'Y':
        if fr == 'C':
            return 'T'
        else:
            return 'C'
    if to == 'S':
        if fr == 'G':
            return 'C'
        else:
            return 'G'
    if to == 'W':
        if fr == 'A':
            return 'T'
        else:
            return 'A'
    if to == 'K':
        if fr == 'G':
            return 'T'
        else:
            return 'G'
    if to == 'M':
        if fr == 'A':
            return 'C'
        else:
            return 'A'

def get_mutation(fr, to):
    if len(to) > 1:
        return to
    elif to == 'A' or to == 'C' or to == 'G' or to == 'T':
        return to
    elif to == '-':
        return ''
    else:
        return get_base_from_iupac(fr, to)

REFERENCE_FILE = sys.argv[1]
MUTATIONS_FILE = sys.argv[2]

chromosomes_ref = {}
full_labels_ref = {}
label = ''
for line in open(REFERENCE_FILE, 'r'):
    if re.match('>', line):
        label = re.match('(.*?)\s', line[1:]).group()[:-1]
        full_labels_ref[label] = line
    else:
        chromosomes_ref[label] = { 1: line, 2: line }

LABEL_IDX = 0
MUTATION_BASE_IDX = 1
MUTATION_FROM_IDX = 2
MUTATION_TO_IDX = 3
MUTATION_PLOIDY_IDX = 4

label = ''
append_start = { 1: 0, 2: 0 }
chromosomes_mutated = { 1: [], 2: [] }

ref_parts = REFERENCE_FILE.split('/')[-1].split('.')
out = open('.'.join(ref_parts[:-1]) + '.mutated.' + ref_parts[-1], 'w', newline='')
for line in open(MUTATIONS_FILE, 'r'):
    fields = line.split()
    if (fields[LABEL_IDX] != label):
        if label:
            finalise_chromosome(chromosomes_mutated[1], chromosomes_mutated[2], chromosomes_ref[label][1][append_start[1]:], chromosomes_ref[label][2][append_start[2]:], full_labels_ref[label], out)
            append_start = { 1: 0, 2: 0 }
            chromosomes_mutated = { 1: [], 2: [] }
        label = fields[LABEL_IDX]
    homologue = int(fields[MUTATION_PLOIDY_IDX])
    pos = int(fields[MUTATION_BASE_IDX])
    to_bases = get_mutation(fields[MUTATION_FROM_IDX], fields[MUTATION_TO_IDX])
    if homologue == 3:
        chromosomes_mutated[1].append(chromosomes_ref[label][1][append_start[1]:(pos - 1)])
        chromosomes_mutated[2].append(chromosomes_ref[label][2][append_start[2]:(pos - 1)])
        chromosomes_mutated[1].append(to_bases)
        chromosomes_mutated[2].append(to_bases)
        append_start[1] = pos
        append_start[2] = pos
    else:
        chromosomes_mutated[homologue].append(chromosomes_ref[label][homologue][append_start[homologue]:(pos - 1)])
        chromosomes_mutated[homologue].append(to_bases)
        append_start[homologue] = pos
finalise_chromosome(chromosomes_mutated[1], chromosomes_mutated[2], chromosomes_ref[label][1][append_start[1]:], chromosomes_ref[label][2][append_start[2]:], full_labels_ref[label], out)
out.close()

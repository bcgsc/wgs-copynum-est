import copy
import re
import sys

def init_chromosome():
    return []

def ploidise(val):
    if HAPLOID:
        return val
    return { 1: val, 2: copy.copy(val) }

def finalise_haploid(mutated, ref, full_label, out):
    mutated.append(ref)
    out.write(full_label)
    out.write(''.join(mutated))

def finalise_chromosome(mutated, ref, start, full_label, out):
    if HAPLOID:
        finalise_haploid(mutated, ref[start:], full_label, out)
    else:
        finalise_haploid(mutated[1], ref[1][start[1]:], full_label, out)
        finalise_haploid(mutated[2], ref[2][start[2]:], full_label, out)

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

def append_haploid(mutated, ref, start, end, to_bases):
    mutated.append(ref[start:end])
    mutated.append(to_bases)

def append_mutation(mutated, ref, start, end, to_bases, copies):
    if copies:
        for i in copies:
            append_haploid(mutated[i], ref[i], start[i], end, to_bases)
        return
    append_haploid(mutated, ref, start, end, to_bases)


HAPLOID = (sys.argv[1] == 'haploid')
REFERENCE_FILE = sys.argv[2]
MUTATIONS_FILE = sys.argv[3]

chromosomes_ref = {}
full_labels_ref = {}
label = ''
for line in open(REFERENCE_FILE, 'r'):
    if re.match('>', line):
        label = re.match('(.*?)\s', line[1:]).group()[:-1]
        full_labels_ref[label] = line
    else:
        chromosomes_ref[label] = ploidise(line)

LABEL_IDX = 0
MUTATION_BASE_IDX = 1
MUTATION_FROM_IDX = 2
MUTATION_TO_IDX = 3
MUTATION_PLOIDY_IDX = 4
APPEND_START_VAL = 0

label = ''
append_start = ploidise(APPEND_START_VAL)
chromosomes_mutated = ploidise(init_chromosome())

ref_parts = REFERENCE_FILE.split('/')[-1].split('.')
out = open('.'.join(ref_parts[:-1]) + '.mutated.' + ref_parts[-1], 'w', newline='')
for line in open(MUTATIONS_FILE, 'r'):
    fields = line.split()
    if (fields[LABEL_IDX] != label):
        if label:
            finalise_chromosome(chromosomes_mutated, chromosomes_ref[label], append_start, full_labels_ref[label], out)
            append_start = ploidise(APPEND_START_VAL)
            chromosomes_mutated = ploidise(init_chromosome())
        label = fields[LABEL_IDX]
    homologue = int(fields[MUTATION_PLOIDY_IDX])
    pos = int(fields[MUTATION_BASE_IDX])
    to_bases = get_mutation(fields[MUTATION_FROM_IDX], fields[MUTATION_TO_IDX])
    is_insertion = (fields[MUTATION_FROM_IDX] == '-')
    copies = []
    if homologue < 3:
        copies = [homologue]
    elif not(HAPLOID):
        copies = [1, 2]
    append_mutation(chromosomes_mutated, chromosomes_ref[label], append_start, pos - 1, to_bases, copies)
    if copies:
        for i in copies:
            append_start[i] = pos - int(is_insertion)
    else:
        append_start = pos - int(is_insertion)
finalise_chromosome(chromosomes_mutated, chromosomes_ref[label], append_start, full_labels_ref[label], out)
out.close()

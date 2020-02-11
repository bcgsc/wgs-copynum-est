#!/bin/bash

set -x

for i in `ls`; do
  output=$(echo $i | perl -ne 'chomp; if (/(.*).hist/) { print "$1.cp.hist"; }')
  awk 'NR > 2 { gsub(/\t/," "); print; }' $i > $output
  mv $output $i
done

set +x

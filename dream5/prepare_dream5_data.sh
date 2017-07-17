#!/bin/bash
# Applies log2 transform to first column followed by removing the last 21 chars of each line

for file in `ls --ignore='*_log*' --ignore='*.sh*'`
do
awk '{print log($1)/log(2)" "$2}' ${file} > ${file}_log2
done

for file in `ls *_log2`
do
rev ${file} | cut -c21- | rev > ${file}_short
done


#! /usr/bin/env python
# -*- coding: utf-8 -*-
'''Prepares a general trainings file with label in first colum

This script does following processing steps:
- Remove offset
- Remove space between bytes
- Removes question marks
- Joins all the lines in the Hex file

Example output:
"
+1 seq..........................
+2 seq..........................
"

Example call:
python createTrainfile.py folde_with_byte_files/ train0Labels.csv train0
'''

import codecs
import sys

if __name__ == "__main__":
    train_folder = sys.argv[1]
    train_labels = codecs.open(sys.argv[2], 'r', 'utf-8')
    outfile = codecs.open(sys.argv[3], 'w', 'utf-8')
    next(train_labels)
    for line in train_labels:
        arr = line.split(',')
        # print(arr)
        if len(arr) != 2:
            continue
        filename = arr[0]
        label = arr[1]
        # print(filename, label)
        try:
            seq = codecs.open(train_folder + "/" + filename + ".bytes")
            outfile.write("+" + label.strip() + " ")
            for line in seq:
                line = line.split(maxsplit=1)[-1]
                line = line.replace(" ", "")
                line = line.replace("?", "")
                outfile.write(line.rstrip())
            outfile.write("\n")
        except FileNotFoundError:
            print("File Not Found: ", filename + ".bytes in " + train_folder)
            continue
        except KeyError:
            print("Catch Key error")
        seq.close()

    outfile.close()
    train_labels.close()

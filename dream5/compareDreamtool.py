#!/usr/bin/env python
"""Module to evaulate the results of SEQL for the DREAM5 challenge"""

from dreamtools import D5C2
import sys

__author__ = "Severin Gsponer"
__copyright__ = "Copyright 2016, Severin Gsponer"
__email__ = "severin.gsponer@insight-centre.org"
__license__ = "GPLv3"


def runDream5Comparison(filename):
    """Saves the comparison table to the DREAM5 teams in a HTML table as well as in a pickle file"""
    s = D5C2()
    s.score(filename)

    table = s.get_table()
    with open('results.html', 'w') as fh:
        fh.write(table.to_html(index=False))

    table.to_pickle('results.p')


if __name__ == "__main__":
    if (len(sys.argv) < 2):
        print("No inputfile.")
        sys.exit(1)
    else:
        runDream5Comparison(str(sys.argv[1]))

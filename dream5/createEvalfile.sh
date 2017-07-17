#!/bin/bash
# This file create a zipfile with the results of DREAM5 prediciton task for upload it to http://www.ebi.ac.uk/saezrodriguez-srv/d5c2/cgi-bin/TF_web.pl
#
# The script expects to have the output of seql_regress of all 66 TF with suffix regress.out in the same folder 
#
# Author: svgspnr (severin.gsponer@insight-centre.org)

# Get directory of the script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cp $DIR/DREAM5_PBM_TeamName_Predictions.txt .

tmpfile=$(mktemp /tmp/dream5CreateEvalfile.XXXXXXX)
for file in `ls *.conc.pred`
do
    # head -n -13 $file |
    awk '{print $2}' ${file} >> $tmpfile
done

paste DREAM5_PBM_TeamName_Predictions.txt $tmpfile | awk 'BEGIN {printf("TF_Id\tArrayType\tSignal_Mean\n")} {printf("%s\t%s\t%f\n",$1,$2,2**$5);}' >DREAM5_PBM_svgsponer_Predictions.txt
gzip DREAM5_PBM_svgsponer_Predictions.txt

rm $tmpfile

#!/bin/bash
# ---------------------------------------------------------------------------
# runDream5Experiment - Runs the full seql regression pipline on the dream5 dataset

# Copyright 2016, Severin Gsponer,,, <svgsponer@svgsponer>
# All rights reserved.

# Usage: runDream5Experiment [-h|--help] [-C C] [-c|--conv convergence threshold][-a|--alpha alpha] [-l|--minpat minpat] [-L|--maxpat maxpat] [-g|--maxgap maxgap] [-G|--maxcongap maxxongap]

# Revision history:
# 2016-06-09 Created by new_script.sh ver. 3.3
# ---------------------------------------------------------------------------

PROGNAME=${0##*/}
VERSION="0.1"

clean_up() { # Perform pre-exit housekeeping
  return
}

error_exit() {
  echo -e "${PROGNAME}: ${1:-"Unknown Error"}" >&2
  clean_up
  exit 1
}

graceful_exit() {
  clean_up
  exit
}

signal_exit() { # Handle trapped signals
  case $1 in
    INT)
      error_exit "Program interrupted by user" ;;
    TERM)
      echo -e "\n$PROGNAME: Program terminated" >&2
      graceful_exit ;;
    *)
      error_exit "$PROGNAME: Terminating on unknown signal" ;;
  esac
}

usage() {
  echo -e "Usage: $PROGNAME [-h|--help] [-C C] [-c|--conv convergence threshold] [-a|--alpha alpha] [-l|--minpat minpat] [-L|--maxpat maxpat] [-g|--maxgap maxgap] [-G|--maxcongap maxcongap] [-W|--workdir workdir] [-O|--outdir outdirname]"
}

help_message() {
  cat <<- _EOF_
  $PROGNAME ver. $VERSION
  Runs the full seql regression pipline on the dream5 dataset

  $(usage)

  Options:
  -h, --help  Display this help message and exit.
  -C, --C  Regularzation weight
  -c, --conv convergence threshold
  -a, --alpha alpha  elastic net weight
  -l, --minpat minpat  min length of any feature
  -L, --maxpat maxpat  max length of any feature
  -g, --maxgap maxgap  number of total wildcards allowed
  -G, --maxcongap maxcongap  number of consecutive wildcards allowed
  -W, --workdir workdir base work dir
  -O, --outdir outputdirname output dir name

_EOF_
  return
}

function runSEQLForSet {
    for file in `ls $PDIR/*_log2_short`
    do
        echo
        BASE=`basename $file`

        #Find TF Name. Switch if work on trainingset
        TF=`echo $file |awk 'BEGIN {FS="_"};{print TF_$(NF-2)}'`
        # TF=`echo $file |awk 'BEGIN {FS="_"};{print $(NF-1)}'`

        printf "Processing File: %s\nTF: %s\n" "${BASE}" "${TF}"

        sqloss_dream -C ${C} -a ${alpha} -c ${conv_t} -v 2 -l ${minpat} -g ${maxgap} -G ${maxcongap} ${file} ${TDIR}/${TSET}_TF_${TF}_log2_short ${WD}/${BASE} /dev/null > ${WD}/${BASE}.learn.out

    done
}

# Trap signals
trap "signal_exit TERM" TERM HUP
trap "signal_exit INT"  INT


# Parse command-line
while [[ -n $1 ]]; do
  case $1 in
    -h | --help)
      help_message; graceful_exit ;;
    -C | --C)
      echo "Regularzation weight"; shift; C="$1" ;;
    -a | --alpha)
      echo "elastic net weight"; shift; alpha="$1" ;;
    -l | --minpat)
      echo "min length of any feature"; shift; minpat="$1" ;;
    -L | --maxpat)
      echo "max length of any feature"; shift; maxpat="$1" ;;
    -g | --maxgap)
      echo "number of total wildcards allowed"; shift; maxgap="$1" ;;
    -G | --maxcongap)
      echo "number of consecutive wildcards allowed"; shift; maxcongap="$1" ;;
    -c | --conv)
      echo "convergence threshold"; shift; conv_t="$1" ;;
    -W | --workdir)
      echo "base work dir"; shift; workdir="$1" ;;
    -O | --outdire)
      echo "output dir"; shift; WDN="$1" ;;
    -* | --*)
      usage
      error_exit "Unknown option $1" ;;
    *)
      echo "Argument $1 to process..." ;;
  esac
  shift
done


#Set default if not set
#C
if [ -z "$C" ]
then
    C=0
fi

# Convergence threshold
if [ -z "$conv_t" ]
then
    conv_t=0.005
fi

#Alpha
if [ -z "$alpha" ]
then
    alpha=0.2
fi

#minpat
if [ -z "$minpat" ]
then
    minpat=0
fi

#maxpat
if [ -z "$maxpat" ]
then
    maxpat=0
fi

#maxgap
if [ -z "$maxgap" ]
then
    maxgap=0
fi

#maxcongap
if [ -z "$maxcongap" ]
then
    maxcongap=0
fi

#Base working dir
if [ -z "$WDB" ]
then
    WDB=`pwd`
fi

#output dir name
if [ -z "$WDN" ]
then
    WDN="minlen${minlen}maxgap${maxgap}maxcongap${maxcongap}conv${conv_t}"
fi

#Check if SEQLBASE is set
if [ -z "$SEQLBASE" ]
then
    printf "\$SEQLBASE has to be set\n"
    graceful_exit
fi

# Main logic

#Create subfolder
WD=${WDB}/${WDN}
mkdir -p ${WD}
cd ${WD}
echo "Change to ${WD}"

#HK->ME
DATADIR=${SEQLBASE}/data/dream5
SET=Predictions_HK_rmFlagged
TSET=Answers_ME
PDIR=${DATADIR}/${SET}
TDIR=${DATADIR}/${TSET}
runSEQLForSet&

#ME->HK
SET=Predictions_ME_rmFlagged
TSET=Answers_HK
PDIR=${DATADIR}/${SET}
TDIR=${DATADIR}/${TSET}
runSEQLForSet&
wait

graceful_exit


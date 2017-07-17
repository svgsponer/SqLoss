# SqLoss: Sequence Regression with Squared Error Loss in All-Subsequence Space
This repository contains all the code to reproduce the experiments in the paper:

**Efficient Sequence Regression by Learning Linear Models in All-Subsequence Space**  
by Severin Gsponer, Barry Smyth and Georgiana Ifrim.


The repository consists of four parts:
* SqLoss: Basic algorithm for linear sequence regression with squared error loss.

* Simulation: Generation of toy sequence regression datasets and evaluation of SqLoss on it.

* Dream5 challenge: Code to run and evaluate SqLoss on the DREAM5 -  Transcription Factor Binding Affinity Challenge.

* MMC: Code to run and evaluate SqLoss on the Microsoft Malware Classification Challenge.

## SqLoss

### Requirements
SqLoss has following dependencies:

* cmake (>=2.8)
* C++ compiler with C++11 support

### Build

* Replace the path place holder in SqLoss/experiments/MMC.cpp.

* To build the SqLoss run these commands in the source (SqLoss) directory:
```bash
mkdir build
cd build
cmake ..
make
```

These commands compile all the files to reproduce the experiments. In particular *sqloss_dream*, *sqloss_regression* and *sqloss_mmc* executable are created.

### Experiments
Here we provide a short overview of the scripts in each folder.
#### Simulated sequence regression:
* generatToySequenceReg.py: Generates sequence regression datasets.
* othermethods/SotA_methods.py: Runs state of the art methods (scikit learn) for comparison.
* eval.py: Extracts results and creates various plots.

#### Dream 5 - TF Challenge:
* prepare_dream5_data.sh: Performs the log2 transformation and cuts sequences as described in the paper.
* run_all_dream.sh: Runs SqLoss for all 66 TFs (sqloss_dream must be in $PATH).
* createEvalFile.sh: Creates submission file as expected of DREAMtools.
* compareDreamtool.py: Evaluates results with the [DREAMtools](https://github.com/dreamtools/dreamtools) (external dependency) 

Data is available [http://dreamchallenges.org/project/dream-5-tf-dna-motif-recognition-challenge/](here).

#### Microsoft Malware Classification:
* create_kfolds.py: Creates stratifies k-folds (depends on sklearn) based on label file provided by Kaggle.
* createTrainfile.py: Creates input file for SqLoss from individual malware files and k-fold label files.
* eval.py: Evaluates obtained results.

Data and further information is available [https://www.kaggle.com/c/malware-classification](here).


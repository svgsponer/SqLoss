/*
 * Author: Georgiana Ifrim (georgiana.ifrim@gmail.com)
 *
 * This library uses a model stored in a trie
 * for fast classification of a given test set.
 *
 * A customized (tuned) classification threshold can be provided as input to the classifier.
 * The program simply applies a suffix tree model to the test documents for predicting classification labels.
 * Prec, Recall, F1 and Accuracy are reported.
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Library General Public
 * License as published by the Free Software Foundation.
 *
 */
#ifndef SEQL_PREDICTOR_H
#define SEQL_PREDICTOR_H

#include <limits>
#include <vector>
#include <string>
#include <map>
#include <algorithm>
#include <cstdio>
#include <unistd.h>
#include <iostream>
#include <fstream>
#include <iterator>
#include <cmath>
#include "common_string_symbol.h"
#include "darts.h"
#include "sys/time.h"
#include "LinearModel.h"
#include "basic_symbol.h"

struct ClassifierStats {
    unsigned int TP = 0;
    unsigned int FP = 0;
    unsigned int FN = 0;
    unsigned int TN = 0;
};

struct RegressionStats {
    unsigned int numberOfDataPoints = 0;
    double sumY=0;
    double sumX=0;
    double sumSqrdE=0;
    double sumAbsE=0;
    double sumx2=0;
    double sumy2=0;
    double sumxy=0;
};

class SEQLPredictor
{
private:

    std::vector <int>  result;
    std::vector <stx::string_symbol> doc;
    std::map <std::string, double> rules;
    std::map <std::string, int> rules_and_ids;

    std::ostream& printRules (std::ostream &os);
    std::ostream& printIds (std::ostream &os);

    bool userule = false;
    int oov_docs {0};

    int verbose {0};
    int token_type {0};

    LinearModel::LinearModel* model;
    vector<pair<double, double> > scores {};
    ClassifierStats stats {};
    RegressionStats reg_stats {};

    void project (std::string prefix,
                  unsigned int pos,
                  size_t trie_pos,
                  size_t str_pos,
                  bool token_type);

    int getOOVDocs();


public:
    SEQLPredictor(int verbose, int token_type, LinearModel::LinearModel* model):
        verbose{verbose}, token_type{token_type},  model{model} {};

    void set_rule(bool t);

    void print_class_stats(std::string filename = "");
    void print_reg_stats(std::string filename = "");

    double predict (const char *line, bool token_type);
    void evalFile(std::string filename, std::string outFileRequested="", int classNumber = -1);
    void tune(std::string filename);
};

namespace SEQLStats{
    // Compute the area under the ROC curve.
    double calcROC(const std::vector< std::pair<double, double> >& forROC );

    // Compute the area under the ROC50 curve.
    // Fixes the number of negatives to 50.
    // Stop computing curve after seeing 50 negatives.
    double calcROC50(const std::vector< std::pair<double, double> >& forROC );
    //Compute r-absolute error
    double calcRabs(const std::vector< std::pair<double, double> >& scores, const double meanY);
    //Compute r-squared score
    double calcR2(const std::vector< std::pair<double, double> >& scores, const double meanY);

}
#endif

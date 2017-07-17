/*
 * Author: Georgiana Ifrim (georgiana.ifrim@gmail.com)
 * SEQL: Sequence Learner
 * This library trains ElasticNet-regularized Logistic Regression and L2-loss (squared-hinge-loss) SVM for Classifying Sequences in the feature space of all possible
 * subsequences in the given training set.
 * Elastic Net regularizer: alpha * L1 + (1 - alpha) * L2, which combines L1 and L2 penalty effects. L1 influences the sparsity of the model, L2 corrects potentially high
 * coeficients resulting due to feature correlation (see Regularization Paths for Generalized Linear Models via Coordinate Descent, by Friedman et al, 2010).
 *
 * License:
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Library General Public
 * License as published by the Free Software Foundation.
 *
 */


#ifndef SEQL_LEARN_H
#define SEQL_LEARN_H

#include <cfloat>
#include <iostream>
#include <cmath>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <algorithm>
#include <numeric>
#include <functional>
#include <map>
#include <set>
#include <iterator>
#include <cstdlib>
#include <cstring>
#include <unistd.h>
// #include <ctime>
#include <memory>
#include <chrono>
#include "sys/time.h"
#include <list>
#include "SNode.h"
#include "CSVwriter.h"
#include "LinearModel.h"
#include "seql_predictor.h"

using namespace std;

class SeqLearner {

private:
    enum Lossfunction { SLR, /* l1SVM, */ l2SVM = 2, SqrdL = 3 };
    // Best ngram rule.
    struct rule_t {
        // Gradient value for this ngram.
        double gradient {0};
        // Length of ngram.
        unsigned int size {0};
        // Ngram label.
        std::string  ngram;
        // Ngram support, e.g. docids where it occurs in the collection.
        std::vector <unsigned int> loc;
        friend bool operator < (const rule_t &r1, const rule_t &r2)
        {
            return r1.ngram < r2.ngram;
        }
        rule_t() = default;
    };
    struct bound_t{
        double gradient;
        double upos;
        double uneg;
        unsigned int support;
    };
    // Entire collection of documents, each doc represented as a string.
    // The collection is a vector of strings.
    const std::vector<string> transaction;
    // True classes.
    const std::vector<double> y;
    // Per document, sum of best beta weights, beta^t * xi = sum_{j best beta coord} gradient_j
    std::vector<double> sum_best_beta;
    // The scalar product obtained with the optimal beta according to the line search for best step size.
    std::vector<double> sum_best_beta_opt;
    // The fraction: 1 / 1 + exp^(yi*beta^t*xi) in the gradient computation.
    std::vector<long double>  exp_fraction;
    vector<double> gradients;
    // Regularized loss function: loss + C * elasticnet_reg
    // SLR loss function: log(1+exp(-yi*beta^t*xi))
    // Squared Hinge SVM loss function: sum_{i|1-yi*beta^t*xi > 0} (1 - yi*beta^t*xi)^2
    // Squared error: (yi-beta^t*xi)^2
    long double loss; // loss without regulatization added
    long double regLoss; // keep loss with regulatization added
    long double old_regLoss; //keep loss in previous iteration for checking convergence

    std::map<string, double> features_cache;
    map<string, double>::iterator features_it;

    // PARAMETERS
    // Objective function. For now choice between logistic regression, l2 (Squared Hinge Loss) and squared error loss.
    Lossfunction objective = SLR;
    // Regularizer value.
    double C = 1;
    // Weight on L1 vs L2 regularizer.
    double alpha = 0.2;
    // Max length for an ngram.
    unsigned int maxpat = 1;
    // Min length for an ngram.
    unsigned int minpat = 0;
    // Min suport for an ngram.
    unsigned int minsup = 1;

    // The sum of squared values of all non-zero beta_j.
    double sum_squared_betas = 0;

    // The sum of abs values of all non-zero beta_j.
    double sum_abs_betas = 0;

    std::set <string> single_node_minsup_cache;

    // Current suboptimal gradient.
    double       tau = 0;

    // Total number of times the pruning condition is checked
    unsigned int total;
    // Total number of times the pruning condition is satisfied.
    unsigned int pruned;
    // Total number of times the best rule is updated.
    unsigned int rewritten;

    // Convergence threshold on aggregated change in score predictions.
    // Used to automatically set the number of optimisation iterations.
    double convergence_threshold = 0.005;

    // Verbosity level: 0 - print no information,
    //                  1 - print profiling information,
    //                  2 - print statistics on model and obj-fct-value per iteration
    //                  > 2 - print details about search for best n-gram and pruning process
    int verbosity = 1;

    // Traversal strategy: BFS or DFS.
    bool traversal_strategy;

    // Profiling variables.
    struct timeval t;
    struct timeval t_origin;
    struct timeval t_start_iter;

    CSVwriter* logger;
    // Bool if csvfile loggin is turned on or not
    bool csvLog;
    //long double LDBL_MAX = numeric_limits<long double>::max();

    // Bool if warmstart should be used. Wamrstart reevaluates the top_nodes found in previous iteration befor
    // start the search at the unigrams.
    bool warmstart {false};

    // Read the input training documents, "true_class document" per line.
    // A line in the training file can be: "+1 a b c"
    bool read (const char *filename);

    // For current ngram, compute the gradient value and check prunning conditions.
    // Update the current optimal ngram.
    /* bool can_prune_and_update_rule (rule_t& rule, SNode *space, unsigned int size); */

    //calculates the gradient of a node
    bound_t calculate_bound(SNode *space);
    //calculates the gradient for each document;
    void calc_doc_gradients();
    // Chechs if node with given gradient can be pruned
    bool can_prune(SNode *space, bound_t bound);
    // Udates the rule if gradient is bigger than tau
    void update_rule(rule_t& rule, SNode* space, unsigned int size, bound_t bound);

    // Try to grow the ngram to next level, and prune the appropriate extensions.
    // The growth is done breadth-first, e.g. grow all unigrams to bi-grams, than all bi-grams to tri-grams, etc.
    void span_bfs (rule_t& rule,
                   SNode *space,
                   std::vector<SNode *>& new_space,
                   unsigned int size);

    std::map<std::string, SNode> find_children(SNode* space);


    // Try to grow the ngram to next level, and prune the appropriate extensions.
    // The growth is done deapth-first rather than breadth-first, e.g. grow each candidate to its longest unpruned sequence
    void span_dfs (rule_t& rule, SNode *space, unsigned int size);

    // Line search method. Search for step size that minimizes loss.
    // Compute loss in middle point of range, beta_n1, and
    // for mid of both ranges beta_n0, beta_n1 and bet_n1, beta_n2
    // Compare the loss for the 3 points, and choose range of 3 points
    // which contains the minimum. Repeat until the range spanned by the 3 points is small enough,
    // e.g. the range approximates well the vector where the loss function is minimized.
    // Return the middle point of the best range.
    void find_best_range(vector<double>& sum_best_beta_n0,
                         vector<double>& sum_best_beta_n1,
                         vector<double>& sum_best_beta_n2,
                         const rule_t& rule,
                         vector<double>& sum_best_beta_opt,
                         const bool is_intercept); // end find_best_range().

    // Line search method. Binary search for optimal step size. Calls find_best_range(...).
    // sum_best_beta keeps track of the scalar product beta_best^t*xi for each doc xi.
    // Instead of working with the new weight vector beta_n+1 obtained as beta_n - epsilon * gradient(beta_n)
    // we work directly with the scalar product.
    // We output the sum_best_beta_opt which contains the scalar poduct of the optimal beta found, by searching for the optimal
    // epsilon, e.g. beta_n+1 = beta_n - epsilon_opt * gradient(beta_n)
    // epsilon is the starting value
    // rule contains info about the gradient at the current iteration
    void binary_line_search(const rule_t& rule, vector<double>& sum_best_beta_opt, const bool is_intercept);

    // Searches the space of all subsequences for the ngram with the ngram with the maximal abolute gradient and saves it in rule
    rule_t findBestNgram(rule_t& rule ,std::vector <SNode*>& old_space, std::vector<SNode*>& new_space, std::map<string, SNode>& seed);

    // Function that calculates the the excat step size for given coordinate.
    // Only used for Squared error loss.
    double excact_step_length(const rule_t& rule, vector<double>& sum_best_beta_opt);

    void warm_start(rule_t& rule);
    vector<SNode*> top_nodes {};

public:
    SeqLearner (std::vector<string> x,
                std::vector<double> y,
                unsigned int objective,
                unsigned int maxpat,
                unsigned int minpat,
                unsigned int minsup,
                unsigned int maxgap,
                unsigned int maxcongap,
                bool token_type,
                bool traversal_strategy,
                double convergence_threshold,
                double regularizer_value,
                double l1vsl2_regularizer,
                int verbosity,
                bool csvLog,
                bool warmstart = false):
    transaction{x}, y{y},
    sum_best_beta(transaction.size(), 0),
    sum_best_beta_opt(transaction.size(), 0),
    exp_fraction(transaction.size(), 1.0/2.0),
    gradients(transaction.size(), 0),
    objective{static_cast<Lossfunction> (objective)},
    C{regularizer_value},
    alpha{l1vsl2_regularizer},
    maxpat{maxpat},
    minpat{minpat},
    minsup{minsup},
    convergence_threshold{convergence_threshold},
    verbosity{verbosity},
    traversal_strategy{traversal_strategy},
    csvLog{csvLog},
    warmstart{warmstart}
    {
        SNode::setupWildcardConstraint(maxgap, maxcongap);
        SNode::tokenType = token_type;
    };

    LinearModel::LinearModel learn (unsigned int maxIter, std::string out,
                                    std::string csvFile);
    LinearModel::LinearModel learn (unsigned int maxIter, std::string out,
                                    std::string csvFile, std::map<std::string, SNode>& seed);
    LinearModel::LinearModel learn (unsigned int maxIter, std::string out, double mean,
                                    std::vector<string> xval, std::vector<double> yval,
                                    std::string csvFile);
    LinearModel::LinearModel learn (unsigned int maxIter, std::string out, double mean,
                                    std::vector<string> xval, std::vector<double> yval,
                                    std::string csvFile, std::map<std::string, SNode>& seed);

    void prepareInvertedIndex (std::map<string, SNode>& seed) ;

    void deleteUndersupportedUnigrams(std::map<string, SNode>& seed);
    double adjust_intercept(vector<double>& sum_best_beta);
    double calc_intercept_gradient(std::vector<double>& sum_best_beta);

    long double computeLossTerm(const double& beta, const double &y);
    long double computeLossTerm(const double& beta, const double &y, long double &exp_fraction);
    // Updates terms of loss function that chagned. vector<> loc contains documnets which loss functions changed
    void updateLoss(long double &loss,
                    const std::vector<double>& new_beta,
                    const std::vector<double>& old_beta,
                    const std::vector<unsigned int> loc);

    double computeLoss(const std::vector<double>& predictions,
                     const std::vector<double>& y_vec
                     );


    // Updates terms of loss function that chagned. vector<> loc contains documnets which loss functions changed
    void updateLoss(long double &loss,
                    const std::vector<double>& new_beta,
                    const std::vector<double>& old_beta,
                    const std::vector<unsigned int> loc,
                    double &sum_abs_scalar_prod_diff,
                    double &sum_abs_scalar_prod,
                    std::vector<double long>& exp_fraction);

    void setup( string csvFile);

};
#endif

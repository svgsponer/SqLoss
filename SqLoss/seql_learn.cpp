/*
 * Author: Severin Gsponer(svgsponer@gamil.com), Georgiana Ifrim (georgiana.ifrim@gmail.com), 
 * License:
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Library General Public
 * License as published by the Free Software Foundation.
 *
 */


#include "seql_learn.h"

using namespace std;

SeqLearner::bound_t SeqLearner::calculate_bound(SNode *space){
    ++total;

    // Upper bound for the positive class.
    double upos = 0;
    // Upper bound for the negative class.
    double uneg = 0;
    // Gradient value at current ngram.
    double gradient = 0;
    // Support of current ngram.
    unsigned int support = 0;
    //string reversed_ngram;
    std::vector <int>& loc = space->loc;
    // Compute the gradient and the upper bound on gradient of extensions.

    for (unsigned int i = 0; i < loc.size(); ++i) {
        if (loc[i] >= 0) continue;
        ++support;
        unsigned int j = (unsigned int)(-loc[i]) - 1;

        switch (objective){

        case SLR:
            // From differentiation we get a - in front of the sum_i_to_N
            // gradient -= y[j] * exp_fraction[j];
            gradient -= gradients[j];

            if (y[j] > 0) {
                upos -= gradients[j];
            } else {
                uneg -= gradients[j];
            }
            break;

        case l2SVM:
            if (gradients[j] > 0) {
                gradient -=  y[j] * gradients[j];

                if (y[j] > 0) {
                    upos -= y[j] * gradients[j];
                } else {
                    uneg -= y[j] * gradients[j];
                }
            }
            break;

        case SqrdL:
            gradient -= gradients[j];
            if (gradients[j] > 0) {
                upos -= gradients[j];
            } else {
                uneg -= gradients[j];
            }
            break;
        }
    }
    // Correct for already selected features
    if(C != 0){

        string ngram = space->getNgram();

        if (verbosity > 3) {
            cout << "\n\ncurrent ngram rule: " << ngram;
            cout << "\nlocation size: " << space->loc.size();
            cout << "\ngradient (before regularizer): " << gradient;
            cout << "\nupos (before regularizer): " << upos;
            cout << "\nuneg (before regularizer): " << uneg;
            cout << "\ntau: " << tau << '\n';
        }

        double current_upos = 0;
        double current_uneg = 0;

        // Retrieve the beta_ij coeficient of this feature. If beta_ij is non-zero,
        // update the gradient: gradient += C * [alpha*sign(beta_j) + (1-alpha)*beta_j];
        // Fct lower_bound return an iterator to the key >= given key.
        features_it = features_cache.lower_bound(ngram);
        // If there are keys starting with this prefix (this ngram starts at pos 0 in existing feature).
        if (features_it != features_cache.end() && features_it->first.find(ngram) == 0) {
            // If found an exact match for the key.
            // add regularizer to gradient.
            if (features_it->first.compare(ngram) == 0) {
                int sign = abs(features_it->second)/features_it->second;
                gradient += C * (alpha * sign + (1-alpha) * features_it->second);
            }

            if (verbosity > 3) {
                cout << "\ngradient after regularizer: " << gradient;
            }
            // Check if current feature s_j is a prefix of any non-zero features s_j'.
            // Check exact bound for every such non-zero feature.
            while (features_it != features_cache.end() && features_it->first.find(ngram) == 0) {
                int sign = abs(features_it->second)/features_it->second;
                current_upos = upos + C * (alpha * sign + (1-alpha) * features_it->second);
                current_uneg = uneg + C * (alpha * sign + (1-alpha) * features_it->second);

                if (verbosity > 3) {
                    cout << "\nexisting feature starting with current ngram rule prefix: "
                         << features_it->first << ", " << features_it->second << ",  sign: " << sign;

                    cout << "\ncurrent_upos: " << current_upos;
                    cout << "\ncurrent_uneg: " << current_uneg;
                    cout << "\ntau: " << tau;
                }
                // Check bound. If any non-zero feature starting with current ngram as a prefix
                // can still qualify for selection in the model,
                // we cannot prune the search space.
                 if (std::max (abs(current_upos), abs(current_uneg)) > tau ) {
                    upos = current_upos;
                    uneg = current_uneg;
                    break;
                }
                ++features_it;
            }
        }
    }
    bound_t bound = {gradient, upos, uneg, support};
    return bound;
}

bool SeqLearner::can_prune(SNode *space, bound_t bound) {
    if (std::max (abs(bound.upos), abs(bound.uneg)) <= tau ) {
        ++pruned;
        if (verbosity > 3) {
            cout << "\n" << space->ngram << ": Pruned due to bound!"
                 << "\n\tgradient: " << bound.gradient
                 << "\n\ttau: " << tau
                 << "\n\tupos: " << bound.upos
                 << "\n\tuneg: " << bound.uneg
                 << "\n";
        }
        return true;
    }
    // Check if support of ngram is below minsupport
    if (bound.support < minsup) {
        ++pruned;
        if (verbosity > 3) {
            cout << "\n" << space->ngram << ": Pruned since support < minsup!\n";
        }
        return true;
    }
    return false;
}
void SeqLearner::update_rule(rule_t& rule, SNode* space, unsigned int size, bound_t bound){
    double g = std::abs (bound.gradient);
    // If current ngram better than previous best ngram, update optimal ngram.
    // Check min length requirement.
    if (g > tau && size >= minpat) {
        ++rewritten;
        top_nodes.push_back(space);
        tau = g;
        rule.gradient = bound.gradient;
        rule.size = size;
        rule.ngram = space->getNgram();

        if (verbosity >= 3) {
            cout << "\n\nnew current best ngram rule: " << space->getNgram();
            cout << "\ngradient: " << bound.gradient << "\n";
        }

        rule.loc.clear ();
        for (unsigned int i = 0; i < space->loc.size(); ++i) {
            // Keep the doc ids where the best ngram appears.
            if (space->loc[i] < 0) rule.loc.push_back ((unsigned int)(-space->loc[i]) - 1);
        }
    }
}

// Try to grow the ngram to next level, and prune the appropriate extensions.
// The growth is done breadth-first, e.g. grow all unigrams to bi-grams, than all bi-grams to tri-grams, etc.
void SeqLearner::span_bfs (rule_t& rule,
                           SNode *space,
                           std::vector<SNode *>& new_space,
                           unsigned int size)
{

    std::vector <SNode *>& next = space->next;
    // If working with gaps.
    // Check if number of consecutive gaps exceeds the max allowed.
    if(SNode::hasWildcardConstraints){
        if(space->violateWildcardConstraint()) return;
    }

    if (!(next.size() == 1 && next[0] == 0)) {
        // If there are candidates for extension, iterate through them, and try to prune some.
        if (! next.empty()) {
            if (verbosity > 4)
                cout << "\n !next.empty()";
            for (auto const &next_space : next) {
                // If the last token is a gap, skip checking gradient and pruning bound, since this is the same as for the prev ngram without the gap token.
                // E.g., if we checked the gradient and bounds for "a" and didnt prune it, then the gradient and bounds for "a*" will be the same,
                // so we can safely skip recomputing the gradient and bounds.
                if (next_space->ne.compare("*") == 0) {
                    if (verbosity > 4)
                        cout << "\nFeature ends in *, skipping gradient and bound computation.";
                    new_space.push_back (next_space);
                } else {
                    bound_t bound = calculate_bound(next_space);
                    if (!can_prune(next_space, bound)) {
                        update_rule(rule, next_space, size, bound);
                        new_space.push_back (next_space);
                    }
                }
            }
        } else {

            // Candidates obtained by extension.
            std::map<string, SNode> candidates = find_children(space);

            space->shrink ();
            if (candidates.empty()){
                next.push_back (0);
            }else{
                next.clear();
                next.reserve(candidates.size());
                // Prepare the candidate extensions.
                for (auto const &currCandidate : candidates) {
                    auto c = new SNode;
                    c->loc = currCandidate.second.loc;
                    c->ne    = currCandidate.first;
                    c->ngram = space->getNgram() + currCandidate.first;
                    c->prev  = space;
                    c->next.clear ();

                    // Keep all the extensions of the current feature for future iterations.
                    // If we need to save memory we could sacrifice this storage.
                    next.push_back(c);

                    // If the last token is a gap, skip checking gradient and pruning bound, since this is the same as for ngram without last gap token.
                    // E.g., if we checked the gradient and bounds for "a" and didnt prune it, then the gradient and bounds for "a*" will be the same,
                    // so we can safely skip recomputing the gradient and bounds.
                    if (c->ne.compare("*") == 0) {
                        if (verbosity > 3)
                            cout << "\nFeature ends in *, skipping gradient and bound computation. Extending to next bfs level.";
                        new_space.push_back (c);
                    } else {
                        bound_t bound = calculate_bound(c);
                        if (! can_prune(c, bound)) {
                            update_rule(rule, c, size, bound);
                            new_space.push_back (c);
                        }
                    }
                }
            }
            // Adjust capacity of next vector
            std::vector<SNode *>(next).swap (next);
        } //end generation of candidates when they weren't stored already.
    }
}

std::map<string, SNode> SeqLearner::find_children(SNode* space){
    // Prepare possible extensions.
    unsigned int docid = 0;
    std::map<string, SNode> candidates;

    std::vector <int>& loc = space->loc;

    // Iterate through the inverted index of the current feature.
    for (auto const& currLoc : loc) {
        // If current Location is negative it indicates a document rather than a possition in a document.
        if (currLoc < 0) {
            docid = (unsigned int)(-currLoc) - 1;
            continue;
        }

        // current unigram where extension is done
        string unigram = space->ne;
        if (verbosity > 4) {
            cout << "\ncurrent pos and start char: " <<  currLoc << " " << transaction[docid][currLoc];
            cout << "\ncurrent unigram to be extended (space->ne):" << unigram;
        }
        string next_unigram;
        // If not re-initialized, it should fail.
        unsigned int pos_start_next_unigram = transaction[docid].size();

        if (currLoc + unigram.size() < transaction[docid].size()) { //unigram is not in the end of doc, thus it can be extended.
            if (verbosity > 4) {
                cout << "\npotential extension...";
            }
            if (!SNode::tokenType) { // Word level token.

                // Find the first space after current unigram position.
                unsigned int pos_space = currLoc + unigram.size();
                // Skip over consecutive spaces.
                while ( (pos_space < transaction[docid].size()) && isspace(transaction[docid][pos_space]) ) {
                    pos_space++;
                }
                // Check if doc ends in spaces, rather than a unigram.
                if (pos_space == transaction[docid].size()) {
                    //cout <<"\ndocument with docid" << docid << " ends in (consecutive) space(s)...move to next doc";
                    //std::exit(-1);
                    continue;
                } else {
                    pos_start_next_unigram = pos_space; //stoped at first non-space after consec spaces
                    size_t pos_next_space = transaction[docid].find(' ', pos_start_next_unigram + 1);

                    // Case1: the next unigram is in the end of the doc, no second space found.
                    if (pos_next_space == string::npos) {
                        next_unigram.assign(transaction[docid].substr(pos_start_next_unigram));
                    } else { //Case2: the next unigram is inside the doc.
                        next_unigram.assign(transaction[docid].substr(pos_start_next_unigram, pos_next_space - pos_start_next_unigram));
                    }
                }
            } else { // Char level token. Skip spaces.
                if (!isspace(transaction[docid][currLoc + 1])) {
                    //cout << "\nnext char is not space";
                    next_unigram = transaction[docid][currLoc + 1]; //next unigram is next non-space char
                    pos_start_next_unigram = currLoc + 1;
                } else { // If next char is space.
                    unsigned int pos_space = currLoc + 1;
                    // Skip consecutive spaces.
                    while ((pos_space < transaction[docid].size()) && isspace(transaction[docid][pos_space])) {
                        pos_space++;
                    }
                    // Check if doc ends in space, rather than a unigram.
                    if (pos_space == transaction[docid].size()) {
                        //cout <<"\ndocument with docid" << docid << " ends in (consecutive) space(s)...move to next doc";
                        //std::exit(-1);
                        continue;
                    }
                    /* //disallow using char-tokenization for space separated tokens.
                       else {
                       pos_start_next_unigram = pos_space;
                       //cout << "\nnext next char is not space";
                       next_unigram = transaction[docid][pos_start_next_unigram];
                       } */
                }
            } //end char level token

            if (next_unigram.empty()) {
                cout <<"\nFATAL...in expansion for next_unigram: expansion of current unigram " << unigram << " is empty...exit";
                std::exit(-1);
            }

            if (verbosity > 4) {
                cout << "\nnext_unigram for extension:" << next_unigram;
                cout << "\npos: " <<  pos_start_next_unigram << " " << transaction[docid][pos_start_next_unigram];
            }

            if (minsup == 1 || single_node_minsup_cache.find (next_unigram) != single_node_minsup_cache.end()) {
                candidates[next_unigram].add (docid, pos_start_next_unigram);
            }

            if (SNode::hasWildcardConstraints) {
                // If we allow gaps, we treat a gap as an additional unigram "*".
                // Its inverted index will simply be a copy pf the inverted index of the original features.
                // Example, for original feature "a", we extend to "a *", where inverted index of "a *" is simply
                // a copy of the inverted index of "a", except for positions where "a" is the last char in the doc.
                candidates["*"].add (docid, pos_start_next_unigram);
            }
        } //end generating candidates for the current pos

    } //end iteration through inverted index (docids iand pos) to find candidates
    return candidates;
}


// Try to grow the ngram to next level, and prune the appropriate extensions.
// The growth is done deapth-first rather than breadth-first, e.g. grow each candidate to its longest unpruned sequence
void SeqLearner::span_dfs (rule_t& rule,
                           SNode *space,
                           unsigned int size)
{
    cout<< "WARNING: span_dfs might not work as expected!" << endl;

    std::vector <SNode *>& next = space->next;

    // Check if ngram larger than maxsize allowed.
    if (size > maxpat) return;

    if(SNode::hasWildcardConstraints){
        if(space->violateWildcardConstraint()) return;
    }

    if (!(next.size() == 1 && next[0] == 0)){
        // If the extensions are already computed, iterate through them and check pruning condition.
        if (! next.empty()) {
            if (verbosity >= 3)
                cout << "\n !next.empty()";
            for (std::vector<SNode*>::iterator it = next.begin(); it != next.end(); ++it) {
                if ((*it)->ne.compare("*") == 0) {
                    if (verbosity > 3)
                        cout << "\nFeature ends in *, skipping gradient and bound computation. Extending to next dfs level.";
                    // Expand each candidate DFS wise.
                    span_dfs(rule, *it, size + 1);
                } else {
                    bound_t bound = calculate_bound(*it);
                    if (! can_prune(*it, bound)) {
                        update_rule(rule, *it, size, bound);
                        // Expand each candidate DFS wise.
                        span_dfs(rule, *it, size + 1);
                    }
                }
            }
        } else {

            // Candidates obtained by extension.
            std::map<string, SNode> candidates = find_children(space);

            // Keep only doc_ids for occurrences of current ngram.
            space->shrink ();

            next.reserve(candidates.size());
            next.clear();
            // Prepare the candidate extensions.
            for (auto const &currCandidate : candidates) {

                SNode* c = new SNode;
                c->loc = currCandidate.second.loc;
                std::vector<int>(c->loc).swap(c->loc);
                c->ne    = currCandidate.first;
                c->ngram = space->getNgram() + currCandidate.first;
                c->prev  = space;
                c->next.clear ();

                // Keep all the extensions of the current feature for future iterations.
                // If we need to save memory we could sacrifice this storage.
                next.push_back (c);

                // If the last token is a gap, skip checking gradient and pruning bound, since this is the same as for ngram without last gap token.
                // E.g., if we checked the gradient and bounds for "a" and didnt prune it, then the gradient and bounds for "a*" will be the same,
                // so we can safely skip recomputing the gradient and bounds.
                if (c->ne.compare("*") == 0) {
                    if (verbosity >= 3)
                        cout << "\nFeature ends in *, skipping gradient and bound computation. Extending to next dfs level.";
                    span_dfs(rule, c, size + 1);
                } else {
                    bound_t bound = calculate_bound(c);
                    if (! can_prune(c, bound)) {
                        update_rule(rule, c, size, bound);
                        // Expand each candidate DFS wise.
                        span_dfs(rule, c, size + 1);
                    }
                }
            }


            if (next.empty()) {
                next.push_back (0);
            }
            std::vector<SNode *>(next).swap (next);
        }
    }
}

// Line search method. Search for step size that minimizes loss.
// Compute loss in middle point of range, beta_n1, and
// for mid of both ranges beta_n0, beta_n1 and bet_n1, beta_n2
// Compare the loss for the 3 points, and choose range of 3 points
// which contains the minimum. Repeat until the range spanned by the 3 points is small enough,
// e.g. the range approximates well the vector where the loss function is minimized.
// Return the middle point of the best range.
void SeqLearner::find_best_range(vector<double>& sum_best_beta_n0,
                                 vector<double>& sum_best_beta_n1,
                                 vector<double>& sum_best_beta_n2,
                                 const rule_t& rule,
                                 vector<double>& sum_best_beta_opt,
                                 const bool is_intercept = false) {

    vector<double> sum_best_beta_mid_n0_n1(sum_best_beta.size());
    vector<double> sum_best_beta_mid_n1_n2(sum_best_beta.size());

    double min_range_size = 1e-3;
    double current_range_size = 0;
    int current_interpolation_iter = 0;

    long double loss_mid_n0_n1 = 0;
    long double loss_mid_n1_n2 = 0;
    long double loss_n1 = 0;
    if(is_intercept){
        for (auto docId = 0; docId < transaction.size(); docId++ ) {
            if (verbosity > 4) {
                cout << "\nsum_best_beta_n0[docId]: " << sum_best_beta_n0[docId];
                cout << "\nsum_best_beta_n1[docId]: " << sum_best_beta_n1[docId];
                cout << "\nsum_best_beta_n2[docId]: " << sum_best_beta_n2[docId];
            }
            current_range_size += abs(sum_best_beta_n2[docId] - sum_best_beta_n0[docId]);
        }
    }else{
        for (auto docId:rule.loc) {
            if (verbosity > 4) {
                cout << "\nsum_best_beta_n0[docId]: " << sum_best_beta_n0[docId];
                cout << "\nsum_best_beta_n1[docId]: " << sum_best_beta_n1[docId];
                cout << "\nsum_best_beta_n2[docId]: " << sum_best_beta_n2[docId];
            }
            current_range_size += abs(sum_best_beta_n2[docId] - sum_best_beta_n0[docId]);
        }
    }
    if (verbosity > 3)
        cout << "\ncurrent range size: " << current_range_size;

    double beta_coef_n1 = 0;
    double beta_coef_mid_n0_n1 = 0;
    double beta_coef_mid_n1_n2 = 0;

    if (C != 0 && sum_squared_betas != 0) {
        features_it = features_cache.find(rule.ngram);
    }
    // Start interpolation loop.
    while (current_range_size > min_range_size) {
        if (verbosity > 3)
            cout << "\ncurrent interpolation iteration: " << current_interpolation_iter;

        for (unsigned int i = 0; i < transaction.size();  ++i) { //loop through training samples
            sum_best_beta_mid_n0_n1[i] = (sum_best_beta_n0[i] + sum_best_beta_n1[i]) / 2;
            sum_best_beta_mid_n1_n2[i] = (sum_best_beta_n1[i] + sum_best_beta_n2[i]) / 2;

            if (is_intercept && C != 0) {
                beta_coef_n1 = sum_best_beta_n1[0] - sum_best_beta[0];
                beta_coef_mid_n0_n1 = sum_best_beta_mid_n0_n1[0] - sum_best_beta[0];
                beta_coef_mid_n1_n2 = sum_best_beta_mid_n1_n2[0] - sum_best_beta[0];
            }else if (C != 0){
                beta_coef_n1 = sum_best_beta_n1[rule.loc[0]] - sum_best_beta[rule.loc[0]];
                beta_coef_mid_n0_n1 = sum_best_beta_mid_n0_n1[rule.loc[0]] - sum_best_beta[rule.loc[0]];
                beta_coef_mid_n1_n2 = sum_best_beta_mid_n1_n2[rule.loc[0]] - sum_best_beta[rule.loc[0]];
            }

            if (verbosity > 4) {
                cout << "\nsum_best_beta_mid_n0_n1[i]: " << sum_best_beta_mid_n0_n1[i];
                cout << "\nsum_best_beta_mid_n1_n2[i]: " << sum_best_beta_mid_n1_n2[i];
            }
            loss_n1 += computeLossTerm(sum_best_beta_n1[i], y[i]);
            loss_mid_n0_n1 += computeLossTerm(sum_best_beta_mid_n0_n1[i], y[i]);
            loss_mid_n1_n2 += computeLossTerm(sum_best_beta_mid_n1_n2[i], y[i]);
        } //end loop through training samples.

        if ( C != 0 ) {
            // Add the Elastic Net regularization term.
            // If this is the first ngram selected.
            if (sum_squared_betas == 0) {
                loss_n1 = loss_n1 + C * (alpha * abs(beta_coef_n1) + (1-alpha) * 0.5 * pow(beta_coef_n1, 2));
                loss_mid_n0_n1 = loss_mid_n0_n1 + C * (alpha * abs(beta_coef_mid_n0_n1) + (1-alpha) * 0.5 * pow(beta_coef_mid_n0_n1, 2));
                loss_mid_n1_n2 = loss_mid_n1_n2 + C * (alpha * abs(beta_coef_mid_n1_n2) + (1-alpha) * 0.5 * pow(beta_coef_mid_n1_n2, 2));
            } else {
                // If this feature was not selected before.
                if (features_it == features_cache.end()) {
                    loss_n1 = loss_n1 + C * (alpha * (sum_abs_betas + abs(beta_coef_n1)) + (1 - alpha) * 0.5 * (sum_squared_betas + pow(beta_coef_n1, 2)));
                    loss_mid_n0_n1 = loss_mid_n0_n1 + C * (alpha * (sum_abs_betas + abs(beta_coef_mid_n0_n1)) + (1 - alpha) * 0.5 * (sum_squared_betas + pow(beta_coef_mid_n0_n1, 2)));
                    loss_mid_n1_n2 = loss_mid_n1_n2 + C * (alpha * (sum_abs_betas + abs(beta_coef_mid_n1_n2)) + (1 - alpha) * 0.5 * (sum_squared_betas + pow(beta_coef_mid_n1_n2, 2)));
                } else {
                    double new_beta_coef_n1 = features_it->second + beta_coef_n1;
                    double new_beta_coef_mid_n0_n1 = features_it->second + beta_coef_mid_n0_n1;
                    double new_beta_coef_mid_n1_n2 = features_it->second + beta_coef_mid_n1_n2;
                    loss_n1 = loss_n1  + C * (alpha * (sum_abs_betas - abs(features_it->second) + abs(new_beta_coef_n1)) + (1 - alpha) * 0.5 * (sum_squared_betas - pow(features_it->second, 2) + pow(new_beta_coef_n1, 2)));
                    loss_mid_n0_n1 = loss_mid_n0_n1  + C * (alpha * (sum_abs_betas - abs(features_it->second) + abs(new_beta_coef_mid_n0_n1)) + (1 - alpha) * 0.5 * (sum_squared_betas - pow(features_it->second, 2) + pow(new_beta_coef_mid_n0_n1, 2)));
                    loss_mid_n1_n2 = loss_mid_n1_n2  + C * (alpha * (sum_abs_betas - abs(features_it->second) + abs(new_beta_coef_mid_n1_n2)) + (1 - alpha) * 0.5 * (sum_squared_betas - pow(features_it->second, 2) + pow(new_beta_coef_mid_n1_n2, 2)));
                }
            }
        }// end check C != 0.

        // Focus on the range that contains the minimum of the loss function.
        // Compare the 3 points beta_n, and mid_beta_n-1_n and mid_beta_n_n+1.
        if (loss_n1 <= loss_mid_n0_n1 && loss_n1 <= loss_mid_n1_n2) {
            // Min is in beta_n1.
            if (verbosity > 4) {
                cout << "\nmin is sum_best_beta_n1";
                cout << "\nloss_mid_n0_n1: " << loss_mid_n0_n1;
                cout << "\nloss_n1: " << loss_n1;
                cout << "\nloss_mid_n1_n2: " << loss_mid_n1_n2;
            }
            // Make the beta_n0 be the beta_mid_n0_n1.
            sum_best_beta_n0.assign(sum_best_beta_mid_n0_n1.begin(), sum_best_beta_mid_n0_n1.end());
            // Make the beta_n2 be the beta_mid_n1_n2.
            sum_best_beta_n2.assign(sum_best_beta_mid_n1_n2.begin(), sum_best_beta_mid_n1_n2.end());
        }
        else {
            if (loss_mid_n0_n1 <= loss_n1 && loss_mid_n0_n1 <= loss_mid_n1_n2) {
                // Min is beta_mid_n0_n1.
                if (verbosity > 4) {
                    cout << "\nmin is sum_best_beta_mid_n0_n1";
                    cout << "\nloss_mid_n0_n1: " << loss_mid_n0_n1;
                    cout << "\nloss_n1: " << loss_n1;
                    cout << "\nloss_mid_n1_n2: " << loss_mid_n1_n2;
                }
                // Make the beta_n2 be the beta_n1.
                sum_best_beta_n2.assign(sum_best_beta_n1.begin(), sum_best_beta_n1.end());
                // Make the beta_n1 be the beta_mid_n0_n1.
                sum_best_beta_n1.assign(sum_best_beta_mid_n0_n1.begin(), sum_best_beta_mid_n0_n1.end());
            } else {
                // Min is beta_mid_n1_n2.
                if (verbosity > 4) {
                    cout << "\nmin is sum_best_beta_mid_n1_n2";
                    cout << "\nloss_mid_n0_n1: " << loss_mid_n0_n1;
                    cout << "\nloss_n1: " << loss_n1;
                    cout << "\nloss_mid_n1_n2: " << loss_mid_n1_n2;
                }
                // Make the beta_n0 be the beta_n1.
                sum_best_beta_n0.assign(sum_best_beta_n1.begin(), sum_best_beta_n1.end());
                // Make the beta_n1 be the beta_mid_n1_n2
                sum_best_beta_n1.assign(sum_best_beta_mid_n1_n2.begin(), sum_best_beta_mid_n1_n2.end());
            }
        }

        ++current_interpolation_iter;
        loss_mid_n0_n1 = 0;
        loss_mid_n1_n2 = 0;
        loss_n1 = 0;
        current_range_size = 0;

        if(is_intercept){
            for (auto docId = 0; docId < transaction.size(); docId++ ) {
                if (verbosity > 4) {
                    cout << "\nsum_best_beta_n0[[docId]: " << sum_best_beta_n0[docId];
                    cout << "\nsum_best_beta_n1[[docId]: " << sum_best_beta_n1[docId];
                    cout << "\nsum_best_beta_n2[[docId]: " << sum_best_beta_n2[docId];
                }
                current_range_size += abs(sum_best_beta_n2[docId] - sum_best_beta_n0[docId]);
            }
        }else{
            for (auto docId:rule.loc) {
                if (verbosity > 4) {
                    cout << "\nsum_best_beta_n0[[docId]: " << sum_best_beta_n0[docId];
                    cout << "\nsum_best_beta_n1[[docId]: " << sum_best_beta_n1[docId];
                    cout << "\nsum_best_beta_n2[[docId]: " << sum_best_beta_n2[docId];
                }
                current_range_size += abs(sum_best_beta_n2[docId] - sum_best_beta_n0[docId]);
            }
        }
        if (verbosity > 4) {
            cout << "\ncurrent range size: " << current_range_size;
        }
    } // end while loop.

    // Keep the middle point of the best range.
    if(is_intercept){
        for (auto docId = 0; docId < transaction.size(); docId++ ) {
            sum_best_beta_opt.at(docId) = sum_best_beta_n1[docId];
        }
    }else{
        for (auto docId:rule.loc) {
            sum_best_beta_opt.at(docId) = sum_best_beta_n1[docId];
        }
 }
} // end find_best_range().

// Line search method. Binary search for optimal step size. Calls find_best_range(...).
// sum_best_beta keeps track of the scalar product beta_best^t*xi for each doc xi.
// Instead of working with the new weight vector beta_n+1 obtained as beta_n - epsilon * gradient(beta_n)
// we work directly with the scalar product.
// We output the sum_best_beta_opt which contains the scalar poduct of the optimal beta found, by searching for the optimal
// epsilon, e.g. beta_n+1 = beta_n - epsilon_opt * gradient(beta_n)
// epsilon is the starting value
// rule contains info about the gradient at the current iteration
void SeqLearner::binary_line_search(const rule_t& rule,
                                    vector<double>& sum_best_beta_opt,
                                    const bool is_intercept = false) {

    // sum_best_beta_opt->clear();
    // Starting value for parameter in step size search.
    // Set the initial epsilon value small enough to guaranteee
    // log-like increases in the first steps.
    double exponent = ceil(log10(abs(rule.gradient)));
    double epsilon = min(1e-3, pow(10, -exponent));

    if (verbosity > 3) {
        cout << "\nrule.ngram: " << rule.ngram;
        cout << "\nrule.gradient: " << rule.gradient;
        cout << "\nexponent of epsilon: " << -exponent;
        cout << "\nepsilon: " << epsilon;
    }

    // Keep track of scalar product at points beta_n-1, beta_n and beta_n+1.
    // They are denoted with beta_n0, beta_n1, beta_n2.
    vector<double> sum_best_beta_n0(sum_best_beta);
    vector<double> sum_best_beta_n1(sum_best_beta);
    vector<double> sum_best_beta_n2(sum_best_beta);

    // Keep track of loss at the three points, n0, n1, n2.
    long double loss_n0 = 0;
    long double loss_n1 = 0;
    long double loss_n2 = loss;
    // added regularization term to loss
    long double regLoss_n2 = loss;
    // Binary search for epsilon. Similar to bracketing phase in which
    // we search for some range with promising epsilon.
    // The second stage finds the epsilon or corresponding weight vector with smallest l2-loss value.

    // **************************************************************************/
    // As long as the l2-loss decreases, double the epsilon.
    // Keep track of the last three values of beta, or correspondingly
    // the last 3 values for the scalar product of beta and xi.
    int n = 0;

    if ( C != 0 && sum_squared_betas != 0) {
        features_it = features_cache.find(rule.ngram);
    }

    double beta_coeficient_update = 0;
    do {
        if (verbosity > 3)
            cout << "\nn: " << n;

        // For each location (e.g. docid), update the score of the documents containing best rule.
        // E.g. update beta^t * xi.
        beta_coeficient_update -= pow(2, n * 1.0) * epsilon * rule.gradient;
        bool print=true;
        if(is_intercept){
            for (auto docid = 0; docid < transaction.size(); docid++) {
                sum_best_beta_n0[docid] = sum_best_beta_n1[docid];
                sum_best_beta_n1[docid] = sum_best_beta_n2[docid];
                sum_best_beta_n2[docid] = sum_best_beta_n1[docid] - pow(2, n * 1.0) * epsilon * rule.gradient;

                if (verbosity > 4 && print) {
                    cout << "\nsum_best_beta_n0[docid]: " << sum_best_beta_n0[docid];
                    cout << "\nsum_best_beta_n1[docid]: " << sum_best_beta_n1[docid];
                    cout << "\nsum_best_beta_n2[docid]: " << sum_best_beta_n2[docid];
                    print=false;
                }
            }
        }else{
            for (auto docid:rule.loc) {
                sum_best_beta_n0[docid] = sum_best_beta_n1[docid];
                sum_best_beta_n1[docid] = sum_best_beta_n2[docid];
                sum_best_beta_n2[docid] = sum_best_beta_n1[docid] - pow(2, n * 1.0) * epsilon * rule.gradient;

                if (verbosity > 4 && print) {
                    cout << "\nsum_best_beta_n0[docid]: " << sum_best_beta_n0[docid];
                    cout << "\nsum_best_beta_n1[docid]: " << sum_best_beta_n1[docid];
                    cout << "\nsum_best_beta_n2[docid]: " << sum_best_beta_n2[docid];
                    print=false;
                }
            }
        }

        // Compute loss for all 3 values: n-1, n, n+1
        // In the first iteration compute necessary loss.
        if (n == 0) {
            loss_n0 = regLoss;
            loss_n1 = regLoss;
        } else {
            // Update just loss_n2.
            // The loss_n0 and loss_n1 are already computed.
            loss_n0 = loss_n1;
            loss_n1 = regLoss_n2;
        }
        if(is_intercept){
            loss_n2 = computeLoss(sum_best_beta_n2, y);
        }else{
            updateLoss(loss_n2, sum_best_beta_n2,sum_best_beta_n1, rule.loc);
        }

        if (verbosity > 4) {
            cout << "\nloss_n2 before adding regularizer: " << loss_n2;
        }
        // If C != 0, add the L2 regularizer term to the l2-loss.
        // If this is the first ngram selected.
        if ( C != 0 && !is_intercept ) {
            if (sum_squared_betas == 0) {
                regLoss_n2 = loss_n2 + C * (alpha * abs(beta_coeficient_update)
                                            + (1 - alpha) * 0.5 * pow(beta_coeficient_update, 2));

                if (verbosity > 4) {
                    cout << "\nregularizer: " << C * (alpha * abs(beta_coeficient_update)
                                                      + (1 - alpha) * 0.5 * pow(beta_coeficient_update, 2));
                }
            } else {
                // If this feature was not selected before.
                if (features_it == features_cache.end()) {
                    regLoss_n2 = loss_n2 + C * (alpha * (sum_abs_betas + abs(beta_coeficient_update))
                                                + (1 - alpha) * 0.5 * (sum_squared_betas + pow(beta_coeficient_update, 2)));
                } else {
                    double new_beta_coeficient = features_it->second + beta_coeficient_update;
                    regLoss_n2 = loss_n2 + C * (alpha * (sum_abs_betas - abs(features_it->second) + abs(new_beta_coeficient))
                                                + (1 - alpha) * 0.5 * (sum_squared_betas - pow(features_it->second, 2) + pow(new_beta_coeficient, 2)));
                }
            }
        }else{
            regLoss_n2 = loss_n2;
        }// end C != 0.

        if (verbosity > 4) {
            cout << "\nloss_n0: " << loss_n0;
            cout << "\nloss_n1: " << loss_n1;
            cout << "\nloss_n2: " << regLoss_n2;
        }
        ++n;
    } while (regLoss_n2 < loss_n1);
    // **************************************************************************/

    if (verbosity > 3)
        cout << "\nFinished doubling epsilon! The monotonicity loss_n+1 < loss_n is broken!";

    // cout<<"\nsum_best_beta_n0[2]: " << sum_best_beta_n0[2];
    // cout<<"\nsum_best_beta_n1[2]: " << sum_best_beta_n1[2];
    // cout<<"\nsum_best_beta_n2[2]: " << sum_best_beta_n2[2];
    // Search for the beta in the range beta_n-1, beta_mid_n-1_n, beta_n, beta_mid_n_n+1, beta_n+1
    // that minimizes the objective function. It suffices to compare the 3 points beta_mid_n-1_n, beta_n, beta_mid_n_n+1,
    // as the min cannot be achieved at the extrem points of the range.
    // Take the 3 point range containing the point that achieves minimum loss.
    // Repeat until the 3 point range is too small, or a fixed number of iterations is achieved.

    // **************************************************************************/
    vector<double> sum_best_beta_mid_n0_n1(sum_best_beta.size());
    vector<double> sum_best_beta_mid_n1_n2(sum_best_beta.size());

    find_best_range(sum_best_beta_n0,
                    sum_best_beta_n1,
                    sum_best_beta_n2,
                    rule,
                    sum_best_beta_opt,
                    is_intercept);
    // **************************************************************************/
} // end binary_line)search().

// Updated the gradients of each document
inline void SeqLearner::calc_doc_gradients(){
    switch (objective){
    case SLR:
        std::transform(exp_fraction.begin(), exp_fraction.end(),
                       y.begin(), gradients.begin(),
                       [](const double exp, const double y_value){
                           return (y_value * exp);
                       });
        break;
    case l2SVM:
        std::transform(sum_best_beta.begin(),sum_best_beta.end(),
                       y.begin(), gradients.begin(),
                       [](const double betax, const double y_value){
                           return std::max(0.0, 2 * (1 - y_value * betax));
                       });
        break;
    case SqrdL:
        std::transform(sum_best_beta.begin(), sum_best_beta.end(),
                       y.begin(), gradients.begin(),
                       [](const double betax, const double y_value){
                           return  2 * (y_value - betax);
                       });
        break;
    };
};

// Searches the space of all subsequences for the ngram with the ngram with the maximal abolute gradient and saves it in rule
SeqLearner::rule_t SeqLearner::findBestNgram(rule_t& rule,
                                             std::vector <SNode*>& old_space,
                                             std::vector<SNode*>& new_space,
                                             std::map<string, SNode>& seed)
{
    // Reset
    tau = 0;
    rule.ngram = "";
    rule.gradient = 0;
    rule.loc.clear ();
    rule.size = 0;
    old_space.clear ();
    new_space.clear ();
    pruned = total = rewritten = 0;


    calc_doc_gradients();

    if (warmstart){
        warm_start(rule);
    }else{
        top_nodes.clear();
    }

    // Iterate through unigrams.
    for (auto& unigram : seed) {
        bound_t bound = calculate_bound(&unigram.second);
        if (!can_prune(&unigram.second, bound)) {
            update_rule(rule, &unigram.second, 1, bound);
            // Check BFS vs DFS traversal.
            if (!traversal_strategy) {
                old_space.push_back (&unigram.second);
            } else {
                // Traversal is DFS.
                span_dfs (rule, &unigram.second, 2);
            }
        }
    }

    // If BFS traversal.
    if (!traversal_strategy) {
        // Search for best n-gram. Try to extend in a bfs fashion,
        // level per level, e.g., first extend unigrams to bigrams, then bigrams to trigrams, etc.
        //*****************************************************/
        for (unsigned int size = 2; size <= maxpat; ++size) {
            for (auto curspace : old_space) {
                span_bfs (rule, curspace, new_space, size);
            }
            if (new_space.empty()) {
                break;
            }
            old_space = new_space;
            new_space.clear ();
        } // End search for best n-gram.
    } // end check BFS traversal.

    if (verbosity >= 2) {
        cout << "\nfound best ngram! ";
        cout << "\nrule.gradient: " << rule.gradient;
        gettimeofday(&t, NULL);
         cout << " (per iter: " << t.tv_sec - t_start_iter.tv_sec << " seconds; " << (t.tv_sec - t_start_iter.tv_sec) / 60.0 << " minutes; total time:"
             << (t.tv_sec - t_origin.tv_sec) / 60.0 << " minutes)";
    }
    return rule;
};

void SeqLearner::warm_start(rule_t& rule){

    //Calculate gradient and bound of nodes that have rewritten in previous iteration.
    auto top_nodes_old (top_nodes);
    top_nodes.clear();
    for (auto orule:top_nodes_old){
        bound_t bound = calculate_bound(orule);
        if(can_prune(orule, bound)){
            update_rule(rule, orule, orule->ngram.size(), bound);
        }
        // std::cout << "Revaluate Node that rewrote last time: " << orule->getNgram() << std::endl;
    }
};



// Function that calculates the the excat step size for given coordinate.
// Only used for Squared error loss.
double SeqLearner::excact_step_length(const rule_t& rule, vector<double>& sum_best_beta_opt){
    double y_btx=0;
    for ( auto docId:rule.loc){
        y_btx += -1*y[docId] + sum_best_beta[docId];
    }
    auto stepsize = y_btx/(rule.loc.size() * rule.gradient);
    auto step = stepsize * rule.gradient;
    // cout<< "\nEta: " << eta << "\nEta * grad: " << stepsize << endl;

    //Update sum_bets_beta_opt acoording to gradient descent step. 
    for( auto docId: rule.loc){
        // sum_best_beta_opt[docId] = sum_best_beta_opt[docId] - stepsize;
        sum_best_beta_opt[docId] = sum_best_beta[docId] - step;
    }
    return -step;
}

double SeqLearner::calc_intercept_gradient(std::vector<double>& sum_best_beta){

    double gradient = 0.0;
    calc_doc_gradients();
    for (auto j = 0; j < transaction.size() ; j++){
        switch (objective){
        case SLR:
            gradient -= gradients[j];
            break;

        case l2SVM:
            gradient -=  y[j] * gradients[j];
            break;

        case SqrdL:
            gradient -= gradients[j];
            break;
        }
    }
    return gradient;
};

// Changes sum_best_beta and loss implicitly
double SeqLearner::adjust_intercept(vector<double>& sum_best_beta)
{
    // Calc gradient
    double gradient = calc_intercept_gradient(sum_best_beta);

    if (gradient == 0) return 0;

    rule_t int_rule;
    int_rule.gradient = gradient;
    double old_beta = sum_best_beta[0];

    binary_line_search(int_rule, sum_best_beta, true);

    double step = old_beta - sum_best_beta[0];
    double new_loss = computeLoss(sum_best_beta, y);
    if( verbosity > 3){
        std::cout << "Info about adjustment of intercept:\n"
            << "Gradient: " << gradient << '\n'
            << "Step: " << step << '\n'
            << "Old_loss: " << loss << '\n'
            << "New_loss: " << new_loss << '\n';
    }
    loss = new_loss;
    return -step;
};

// Wrapper
LinearModel::LinearModel SeqLearner::learn (unsigned int maxitr, std::string out,
                                            std::string csvFile)
{
    return learn(maxitr, out, 0, std::vector<string> (), std::vector<double> (), csvFile);
 };

LinearModel::LinearModel SeqLearner::learn (unsigned int maxitr, std::string out,
                                            std::string csvFile, std::map<string, SNode>& seed )
{
    return learn(maxitr, out, 0, std::vector<string> (), std::vector<double> (), csvFile, seed);
 };
LinearModel::LinearModel SeqLearner::learn (unsigned int maxitr, std::string out ,
                                            double Y_mean,
                                            std::vector<string> x_val, std::vector<double> y_val,
                                            std::string csvFile)
{
    // A map from unigrams to search_space.
    std::map <string, SNode> seed;
    prepareInvertedIndex(seed);
    return learn(maxitr, out, Y_mean, x_val, y_val, csvFile, seed);
};

LinearModel::LinearModel SeqLearner::learn (unsigned int maxitr, std::string out ,
                                            double Y_mean,
                                            std::vector<string> x_val, std::vector<double> y_val,
                                            std::string csvFile,
                                            std::map<string, SNode>& seed
                                           )
 {
     LinearModel::LinearModel model;
     if (verbosity >= 1) {
         cout << "\nParameters used: " << "obective fct: " << objective << " T: " << maxitr << " minpat: " << minpat << " maxpat: " << maxpat << " minsup: " << minsup
              << " maxgap: " << SNode::totalWildcardLimit << " maxcongap: " << SNode::consecWildcardLimit << " token_type: " << SNode::tokenType << " traversal_strategy: " << traversal_strategy
              << " convergence_threshold: "  << convergence_threshold << " C (regularizer value): " << C << " alpha (weight on l1_vs_l2_regularizer): "
              << alpha  << " verbosity: " << verbosity<< endl;
     }

     gettimeofday(&t_origin, NULL);

     setup(csvFile);



     deleteUndersupportedUnigrams(seed);

     std::vector <SNode*> old_space;
     std::vector <SNode*> new_space;

     // Set bias and initialize sum_best_beta with bias value
     model.set_bias(Y_mean);
     std::fill(sum_best_beta.begin(), sum_best_beta.end(), 0);
     std::fill(sum_best_beta_opt.begin(), sum_best_beta_opt.end(), 0);
     // The optimal step length.
     double step_length_opt;
     // Set the convergence threshold as in paper by Madigan et al on BBR.
     //double convergence_threshold = 0.005;
     double convergence_rate;

     // Current rule.
     rule_t rule;
     double sum_abs_scalar_prod_diff{0};
     double sum_abs_scalar_prod{0};
     // Compute loss with start beta vector.
     loss = computeLoss(sum_best_beta, y);
     regLoss = loss;


     long double int_step{0};
     
     long double validation_loss{0};
     // CSVwriter* validation_logger = new CSVwriter{"validationLog.csv"};
     if(!x_val.empty()){
         //Validation set
         validation_loss = computeLoss(sum_best_beta, y_val);
         // validation_logger->DoLog("itr,validation_loss,loss,reg_loss");
         // validation_logger->DoLog(0, validation_loss, loss, regLoss);
     }
     if (verbosity >= 1) {
         cout << "\nstart loss: " << loss << '\n';
     }
     // Loop for number of given optimization iterations.
     for (unsigned int itr = 0; itr < maxitr; ++itr) {
         /** CONDITION: sum_best_beta and sum_best_beta_opt are equal **/

         gettimeofday(&t_start_iter, NULL);

         // Adjust intercept term. Only implemented for SqLoss
         int_step = adjust_intercept(sum_best_beta);
         model.add(int_step, "*INTERCEPT*");
         sum_best_beta_opt.assign(sum_best_beta.begin(), sum_best_beta.end());

         // Recalculate Σꞵ'x from scratch. There is probably a better solution
         sum_abs_scalar_prod = 0;
         for ( auto doc : sum_best_beta_opt){
             sum_abs_scalar_prod += abs(doc);
         }

         // Search in the feature space for the Ngram with the best absolute gradient value
         findBestNgram(rule, old_space, new_space, seed);
         // rule contains the best best ngram
         // Checck if we found ngram with non zero gradient
         if(rule.loc.size() == 0){
             cout<<"\nBest ngram has a gradient of 0 => Stop search\n";
             break;
         }
         // Optimal stepsize detemination (slr and l2svm by lineseach, SqrdL by excat computation)
         if ( objective != SqrdL){
             // Use line search to detect the best step_length/learning_rate.
             // The method does binary search for the step length, by using a start parameter epsilon.
             // It works by first bracketing a promising value, followed by focusing on the smallest interval containing this value.
             binary_line_search(rule, sum_best_beta_opt);
             // TODO this is not the optimal step length rather optimal change in the betas -eta * gradient
             // The optimal step length as obtained from the line search.
             step_length_opt = sum_best_beta_opt[rule.loc[0]] - sum_best_beta[rule.loc[0]];
             // cout << "\nOptimal step length: " << step_length_opt;
         }else{
             step_length_opt = excact_step_length(rule, sum_best_beta_opt);
         }

         // Update the weight of the best n-gram.
         // Insert or update new feature.
         if ( C != 0 ) {
             // Inserts if not there else return reference to featrue
             auto feature_insert = features_cache.insert({rule.ngram, 0});
            // Adjust coeficient and the sums of coeficients.
             if(!feature_insert.second){
                 sum_squared_betas = sum_squared_betas - pow(feature_insert.first->second, 2);
                 sum_abs_betas = sum_abs_betas - abs(feature_insert.first->second);
             }
             feature_insert.first->second += step_length_opt;
             sum_squared_betas += pow(feature_insert.first->second, 2);
             sum_abs_betas += abs(feature_insert.first->second);

         }

         // Remember the loss from prev iteration.
         old_regLoss = regLoss;
         updateLoss(loss, sum_best_beta_opt, sum_best_beta, rule.loc, sum_abs_scalar_prod_diff, sum_abs_scalar_prod, exp_fraction);

         if (verbosity >= 2) {
             cout << "\nloss: " << loss;
             if ( C != 0 ) {
                 cout << "\npenalty_term: " << C * (alpha * sum_abs_betas + (1 - alpha) * 0.5 * sum_squared_betas);
             }
         }
         // Update the log-likelihood with the regularizer term.
         if ( C != 0 ) {
             regLoss = loss + C * (alpha * sum_abs_betas + (1 - alpha) * 0.5 * sum_squared_betas);
         }else{
             regLoss = loss;
         }

         //stop if loss doesn't improve; a failsafe check on top of conv_rate (based on predicted score residuals) reaching conv_threshold
         if (old_regLoss - regLoss == 0) {
             if (verbosity >= 1) {
                 cout << "\n\nFinish iterations due to: no change in loss value!";
                 cout << "\nloss + penalty term: " << regLoss;
                 cout << "\n# iterations: " << itr + 1;
             }
             break;
         }

         // The optimal step length as obtained from the line search.
         // Stop the alg if weight of best grad feature is below a given threshold.
         // Inspired by paper of Liblinear people that use a thereshold on the value of the gradient to stop close to optimal solution.
         if (abs(step_length_opt) > 1e-8){
             model.add(step_length_opt, rule.ngram);
         } else {
             if (verbosity >= 1) {
                 cout << "\n\nFinish iterations due to: step_length_opt <= 1e-8 (due to numerical precision loss doesn't improve for such small weights)!";
                 cout << "\n# iterations: " << itr + 1;
                 cout << "\nstep_length_opt: " << step_length_opt;
                 cout << "\nngram: " << rule.ngram;
             }
             break;
         }

         if (verbosity >= 2) {
             std::cout <<  "\n #itr: " << itr
                       << " #features: " << features_cache.size()
                       << " #rewrote: " << rewritten
                       << " #prone: " << pruned
                       << " #total: " << total
                       << " stepLength: " << step_length_opt
                       << " rule: " << rule.ngram;

             cout << "\nloss + penalty term: " << regLoss;
             cout.flush();
         }

         // Set the convergence rate as in paper by Madigan et al on BBR.
         convergence_rate = sum_abs_scalar_prod_diff / (1 + sum_abs_scalar_prod);
         // cout << "\nconvergence rate: " << convergence_rate;
         if (convergence_rate <= convergence_threshold) {
             if (verbosity >= 1) {
                 // cout<< "\n\nsum_stats:\n" << sum_abs_scalar_prod_diff << '\n';
                 // cout<<  (1 + sum_abs_scalar_prod);
                 cout << "\nconvergence rate: " << convergence_rate;
                 cout << "\n\nFinish iterations due to: convergence test (convergence_thereshold=" << convergence_threshold << ")!";
                 cout << "\n# iterations: " << itr + 1;
             }
             if (csvLog){
                 logger->DoLog(itr, features_cache.size(),rewritten,pruned,total,step_length_opt,rule.ngram,loss, regLoss,convergence_rate);
             }
             break;
         } // Otherwise, loop up to the user provided # iter or convergence threshold.

         // Validation set evaluation
         if (!x_val.empty() && (itr + 1)% 5 == 0){
             std::cout << "\nstart validation" << std::endl;
             long double old_validation_loss = validation_loss;
             model.build_tree(1);
             model.set_bias(Y_mean);
             SEQLPredictor predictor{1, SNode::tokenType, &model};

             vector<double> predictions (x_val.size(), 0);
             
             std::transform(x_val.begin(), x_val.end(), y_val.begin(), predictions.begin(),
                            [&](string x, double y){
                                return pow(y - predictor.predict(x.c_str(), SNode::tokenType),2);
                            });
             
             long double validation_loss = std::accumulate(predictions.begin(),predictions.end(), 0.0L);

             // validation_logger->DoLog(itr,validation_loss, loss, regLoss);
             if (validation_loss > old_validation_loss) {
                 if (verbosity >= 1) {
                     cout << "\nFinished since validationset loss increased iter:" << itr;
                     cout << "\nold: " << old_validation_loss;
                     cout << "\nnew: " << validation_loss;
                 }
                 break;
             }
         }
         //sum_best_beta_opt is the optimum found using line search
         sum_best_beta.assign(sum_best_beta_opt.begin(), sum_best_beta_opt.end());
         if (csvLog){
             logger->DoLog(itr, features_cache.size(),
                           rewritten, pruned, total,
                           step_length_opt, rule.ngram,
                           loss, regLoss, convergence_rate);
         }
     } //end optimization iterations.

     gettimeofday(&t, NULL);
     if (verbosity >= 1) {
         if ( C != 0 ) {
             cout << "\nend penalty_term: " << C * (alpha * sum_abs_betas + (1 - alpha) * 0.5 * sum_squared_betas);
         }
         cout << "\nend loss + penalty_term: " << regLoss;
         cout << "\n\ntotal time: " << t.tv_sec - t_origin.tv_sec << " seconds; " << (t.tv_sec - t_origin.tv_sec) / 60.0 << " minutes\n ";
     }
     // model.print_fulllist (out);
     return model;
} //end run().

void SeqLearner::prepareInvertedIndex (std::map<string, SNode>& seed) {
    string unigram;
    bool at_space = false;

    // Prepare the locations for unigrams.
    if (verbosity >= 1) {
        cout << "\nprepare inverted index for unigrams";
    }

    for (unsigned int docid = 0; docid < transaction.size(); ++docid) {
        at_space = false;
        //cout << "\nscanning docid: " << docid << ", class y: " << y[docid] << "\n";
        for (unsigned int pos = 0; pos < transaction[docid].size(); ++pos) {
            // Skip white spaces. They are not considered as unigrams.
            if (isspace(transaction[docid][pos])) {
                at_space = true;
                continue;
            }
            // If word level tokens.
            if (!SNode::tokenType) {
                if (at_space) {
                    at_space = false;
                    if (!unigram.empty()) {
                        SNode & tmp = seed[unigram];
                        tmp.add (docid,pos - unigram.size() - 1);
                        tmp.next.clear ();
                        tmp.ne = unigram;
                        tmp.ngram = unigram;
                        tmp.prev = 0;
                        unigram.clear();
                    }
                    unigram.push_back(transaction[docid][pos]);
                } else {
                    unigram.push_back(transaction[docid][pos]);
                }
            } else {
                if (at_space) {
                    //Previous char was a space.
                    //Disallow using char-tokenization for space-separated tokens. It confuses the features, whether to add " " or not.
                    cout << "\nFATAL...found space in docid: " << docid << ", at position: " << pos-1;
                    cout << "\nFATAL...char-tokenization assumes contiguous tokens (i.e., tokens are not separated by space).";
                    cout << "\nFor space separated tokens please use word-tokenization or remove spaces to get valid input for char-tokenization.";
                    cout << "\n...Exiting.....\n";
                    std::exit(-1);
                }
                // Char (i.e. byte) level token.
                unigram = transaction[docid][pos];
                SNode & tmp = seed[unigram];
                tmp.add (docid,pos);
                tmp.next.clear ();
                tmp.ne = unigram;
                tmp.ngram = unigram;
                tmp.prev = 0;
                unigram.clear();
            }
        } //end for transaction.
        // For word-tokens take care of last word of doc.
        if (!SNode::tokenType) {
            if (!unigram.empty()) {
                SNode & tmp = seed[unigram];
                tmp.add (docid, transaction[docid].size() - unigram.size());
                tmp.next.clear ();
                tmp.ne = unigram;
                tmp.ngram = unigram;
                tmp.prev = 0;
                unigram.clear();
            }
        }
    } //end for docid.

};

void SeqLearner::deleteUndersupportedUnigrams(std::map<string, SNode>& seed){
    // Keep only unigrams above minsup threshold.
    for (auto it= seed.cbegin(); it != seed.cend();) {
        if (it->second.support() < minsup) {
            if (verbosity >= 1) {
                cout << "\nremove unigram (minsup):" << it->first;
                cout.flush();
            }
            seed.erase(it++);
        } else {
            single_node_minsup_cache.insert (it->second.ne);
            if (verbosity >= 1) {
                cout << "\ndistinct unigram:" << it->first;
            }
            ++it;
        }
    }
    if( single_node_minsup_cache.size()==0){
        cout << "\n>>> NO UNIGRAM LEFT\nMaybe adjust the minsup parameter";
        exit(1);
    };
    gettimeofday(&t, NULL);
    if (verbosity >= 1) {
        cout << "\n# distinct unigrams: " << single_node_minsup_cache.size();
        cout << " ( " << (t.tv_sec - t_origin.tv_sec) << " seconds; " << (t.tv_sec - t_origin.tv_sec) / 60.0 << " minutes )";
        cout.flush();
    }
};


long double SeqLearner::computeLossTerm(const double& beta, const double &y){
    switch (objective){
    case SLR:
        if (-y * beta > 8000) {
            return log(LDBL_MAX);
        } else {
            return log(1 + exp(-y * beta));
        }
    case l2SVM:
        if (1 - y * beta > 0)
            return pow(1 - y * beta, 2);
        else
            return 0;
    case SqrdL:
        return pow(y - beta, 2);
    default:
        return 0;
    };
}

long double SeqLearner::computeLossTerm(const double& beta, const double& y, long double& exp_fraction){
    switch (objective){
    case SLR:
        if (y * beta > 8000) {
            exp_fraction = 0;
        } else {
            exp_fraction = 1 / (1 + exp(y * beta));
        }
        if (-y * beta > 8000) {
            return log(LDBL_MAX);
        } else {
            return log(1 + exp(-y * beta));
        }
    case l2SVM:
        if (1 - y * beta > 0)
            return pow(1 - y * beta, 2);
        else
            return 0;
    case SqrdL:
        return pow(y - beta, 2);
    default:
        return 0;
    };
}
// Updates terms of loss function that chagned. vector<> loc contains documnets which loss functions changed
void SeqLearner::updateLoss(long double &loss, const std::vector<double>& new_beta, const std::vector<double>& old_beta, const std::vector<unsigned int> loc){
    for (auto docid:loc) {
        loss -= computeLossTerm(old_beta[docid], y[docid]);
        loss += computeLossTerm(new_beta[docid], y[docid]);
    }
}


double SeqLearner::computeLoss(const std::vector<double>& predictions,
                             const std::vector<double>& y_vec){
    double loss = 0;
    for (unsigned int i = 0; i < y_vec.size();  ++i) {
        loss += computeLossTerm(predictions[i], y_vec[i]);
    }
    return loss;
}

// Updates terms of loss function that chagned. vector<> loc contains documnets which loss functions changed
void SeqLearner::updateLoss(long double &loss,  const std::vector<double>& new_beta, const std::vector<double>& old_beta, const std::vector<unsigned int> loc,
                double &sum_abs_scalar_prod_diff, double &sum_abs_scalar_prod, std::vector<double long>& exp_fraction){

    sum_abs_scalar_prod_diff = 0;
    for (auto i:loc) {
        loss -= computeLossTerm(old_beta[i],y[i], exp_fraction[i]);
        loss += computeLossTerm(new_beta[i],y[i], exp_fraction[i]);

        // Compute the sum of per document difference between the scalar product at 2 consecutive iterations.
        sum_abs_scalar_prod_diff += abs(new_beta[i] - sum_best_beta[i]);

        // Compute the sum of per document scalar product at current iteration.
        sum_abs_scalar_prod -= abs(old_beta[i]);
        sum_abs_scalar_prod += abs(new_beta[i]);
    }
}

void SeqLearner::setup(string csvFile){
    if (csvLog){
        logger = new CSVwriter{csvFile};
        logger->DoLog("Iteration,#Features,#rewritten,#pruned,total,optimalStepLength,symbol,loss,regLoss,convRate");
    }
    std::cout.setf(std::ios::fixed,std::ios::floatfield);
    std::cout.precision(8);
}

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
#include "seql_predictor.h"


template <typename T1, typename T2>
struct pair_2nd_cmp: public std::binary_function<bool, T1, T2> {
    bool operator () (const std::pair <T1, T2>& x1, const std::pair<T1, T2> &x2)
    {
        return x1.second > x2.second;
    }
};


void SEQLPredictor::project (std::string prefix,
                  unsigned int pos,
                  size_t trie_pos,
                  size_t str_pos,
                  bool token_type)
    {
        if (pos == doc.size() - 1) return;

        // Check traversal with both the next actual unigram in the doc and the wildcard *.
        string next_unigrams[2];
        next_unigrams[0] = doc[pos + 1].key();
        next_unigrams[1] = "*";

        for (int i = 0; i < 2; ++i) {

            string next_unigram = next_unigrams[i];
            std::string item;
            if (!token_type) { //word-level token
                item = prefix + " " + next_unigram;
            } else { // char-level token
                item = prefix + next_unigram;
            }
            //cout << "\nitem: " << item.c_str();
            size_t new_trie_pos = trie_pos;
            size_t new_str_pos  = str_pos;
            int id = model->da->traverse (item.c_str(), new_trie_pos, new_str_pos);
            //cout <<"\nid: " << id;

            //if (id == -2) return;
            if (id == -2) {
                if (i == 0) continue;
                else return;
            }
            if (id >= 0) {
                if (userule) {
                    //cout << "\nnew rule: " << item;
                    rules.insert (std::make_pair (item, model->weights[id]));
                    rules_and_ids.insert (std::make_pair (item, id));
                }
                result.push_back (id);
            }
            project (item, pos + 1, new_trie_pos, new_str_pos, token_type);
        }
    }


int SEQLPredictor::getOOVDocs() {
        return oov_docs;
    }

void SEQLPredictor::set_rule(bool t)
    {
        userule = t;
    }


double SEQLPredictor::predict (const char *line, bool token_type)
    {
        result.clear ();
        doc.clear ();
        rules.clear ();

        // Prepare instance as a vector of string_symbol
        str2node (line, doc, token_type);

        for (unsigned int i = 0; i < doc.size(); ++i) {
            std::string item = doc[i].key();
            int id;
            model->da->exactMatchSearch (item.c_str(), id);
            //int id = da.exactMatchSearch (doc[i].key().c_str());
            if (id == -2) continue;
            if (id >= 0) {
                if (userule) {
                    rules.insert (std::make_pair (doc[i].key(), model->weights[id]));
                    rules_and_ids.insert (std::make_pair (doc[i].key(), id));
                }
                result.push_back (id);
            }
            project (doc[i].key(), i, 0, 0, token_type);
        }

        std::sort (result.begin(), result.end());

        // Binary frequencies, erase the duplicate feature ids, features count only once.
        result.erase (std::unique (result.begin(), result.end()), result.end());
        if (result.size() == 0) {
            if (userule)
                cout << "\n Test doc out of vocabulary\n";
            oov_docs++;
        }
        // Add up wheight of all found features as well as intercept
        double r = model->get_bias();
        r += model->intercept;
        for (auto res : result) r += model->weights[res];

        return r;
    }

std::ostream& SEQLPredictor::printRules (std::ostream &os)
    {
        std::vector <std::pair <std::string, double> > tmp;

        for (auto it = rules.begin(); it != rules.end(); ++it)
            tmp.push_back (std::make_pair (it->first,  it->second));

        std::sort (tmp.begin(), tmp.end(), pair_2nd_cmp<std::string, double>());
        os << "\nrule:\n" << model->get_bias() << " __MEAN__\n"
           << model->intercept << " *INTERCEPT*" << '\n';

        for (auto it = rules.begin(); it != rules.end(); ++it)
            os << it->first << " " << it->second << '\n';

        os << "Number of rules: " << rules.size() << '\n';
        return os;
    }

std::ostream& SEQLPredictor::printIds (std::ostream &os) {
        for (std::map <std::string, int>::iterator it = rules_and_ids.begin(); it != rules_and_ids.end(); ++it)
            os << (it->second + 1) << ":1.0 ";
        os << "\n";

        return os;
    }


void SEQLPredictor::evalFile(std::string filename, std::string outFileRequested, int classNumber ){
    std::unique_ptr<std::istream> is {nullptr};
    if (filename == "-") is.reset(&std::cin);
    else              is.reset(new std::ifstream(filename.c_str()));

    if (! *is){
        std::cerr << "Error: " << filename << " No such file or directory" << std::endl;
        exit (1);
    }

    std::ofstream realOutFile;
    if(!outFileRequested.empty())
        realOutFile.open(outFileRequested, std::ios::out);
    std::ostream & outFile = (!outFileRequested.empty() ? realOutFile : std::cout);

    std::string line;
    char *column[4];
    // Predicted and true scores for all docs.

    ClassifierStats local_stats;
    scores.clear();
    oov_docs = 0;
    //REGRESSION STATS PART
    RegressionStats local_reg_stats;

    // cout << "\n\nreading test data...\n";
    while (std::getline (*is, line)) {

        if (line[0] == '\0' || line[0] == ';') continue;
        if (line[line.size() - 1] == '\r') {
            line[line.size() - 1] = '\0';
        }
        //cout << "\nline:*" << aux.c_str() << "*";

        if (2 != tokenize ((char *)line.c_str(), "\t ", column, 2)) {
            std::cerr << "Format Error: " << line.c_str() << std::endl;
            exit(1);
        }

        //cout <<"\ncolumn[0]:*" << column[0] << "*";
        //cout <<"\ncolumn[1]:*" << column[1] << "*";
        //cout.flush();
        double y;
        if (classNumber == -1){
            y = atof(column[0]);
        }else{
            y = atoi(column[0]) == classNumber ? +1 : -1;
        }
        double predicted_score = predict (column[1], token_type);

        // Keep predicted and true score.
        scores.push_back(pair<double, double>(predicted_score, y));

        // Transform the predicted_score which is a real number, into a probability,
        // using the logistic transformation: exp^{predicted_score} / 1 + exp^{predicted_score} = 1 / 1 + e^{-predicted_score}.
        double predicted_prob;
        if (predicted_score < -8000) {
            predicted_prob = 0;
        } else {
            predicted_prob = 1.0 / (1.0 + exp(-predicted_score));
        }

        if (verbose == 1) {
            outFile << predicted_score << "\n";
        } else if (verbose == 2) {
            outFile << y << " " << predicted_score << " " << predicted_prob <<  " " << column[1] << "\n";
        } else if (verbose == 4) {
            outFile << "<instance>" << "\n";
            outFile << y << " " << predicted_score << " " << predicted_prob << " " << column[1] << "\n";
            printRules (outFile);
            outFile << "</instance>" << "\n\n";
        } else if (verbose == 5) {
            outFile << y << " ";
            printIds (outFile);
        }
        //REGRESSION STATS UPDATE
        local_reg_stats.sumSqrdE += pow(predicted_score - y,2);
        local_reg_stats.sumAbsE += abs(predicted_score - y);
        local_reg_stats.sumx2 += pow(predicted_score,2);
        local_reg_stats.sumy2 += pow(y,2);
        local_reg_stats.sumxy += predicted_score * y;
        local_reg_stats.numberOfDataPoints++;
        local_reg_stats.sumY += y;
        local_reg_stats.sumX += predicted_score;

        if (predicted_score > 0) {
            if(y > 0) local_stats.TP++; else local_stats.FP++;
        } else {
            if(y > 0) local_stats.FN++; else local_stats.TN++;
        }
    }
    stats = local_stats;
    reg_stats = local_reg_stats;

}

void SEQLPredictor::print_reg_stats(std::string filename){
    std::streambuf * buf;
    std::ofstream of;

    if(!filename.empty()){
        of.open(filename);
        buf = of.rdbuf();
    } else {
        buf = std::cout.rdbuf();
    }

    std::ostream out(buf);
    double meanY = reg_stats.sumY / reg_stats.numberOfDataPoints;
    double meanSqrdE = reg_stats.sumSqrdE / reg_stats.numberOfDataPoints;
    double meanAbsE = reg_stats.sumAbsE / reg_stats.numberOfDataPoints;
    double r2E = SEQLStats::calcR2(scores,meanY);
    double rAbsE = SEQLStats::calcRabs(scores,meanY);
    double pearsonCorr= (reg_stats.sumxy-((reg_stats.sumX*reg_stats.sumY)/reg_stats.numberOfDataPoints)) /
        (sqrt(reg_stats.sumx2-pow(reg_stats.sumX,2)/reg_stats.numberOfDataPoints) * sqrt(reg_stats.sumy2-pow(reg_stats.sumY,2)/reg_stats.numberOfDataPoints));
    //if (verbose >= 3) {
    std::printf ("MeanY:   %.5f\n", meanY);
    std::printf ("Bias:   %.5f\n", model->get_bias());
    std::printf ("mean-squared error:   %f\n", meanSqrdE);
    std::printf ("root mean-squared error:   %f\n",sqrt(meanSqrdE));
    std::printf ("mean-absolute error:   %f\n", meanAbsE);
    std::printf ("relative-squared error:   %f\n", r2E);
    std::printf ("root relative-squared error:   %f\n", sqrt(r2E));
    std::printf ("relative-mean error:   %f\n", rAbsE);
    std::printf ("r-squared error (1-rel.-sqrd.err):   %f\n", 1-r2E);
    std::printf ("pearson correlation:   %f\n", pearsonCorr);
    std::printf ("OOV docs:   %d\n", getOOVDocs());


    if(!filename.empty()){
        out <<	 "{" << '\n';
        out << "\"MeanY\": " << meanY << '\n';
        out << "\"Bias\": " << model->get_bias() << '\n';
        out << "\"mean-squared error\": " << meanSqrdE << '\n';
        out << "\"root mean-squared error\": " << sqrt(meanSqrdE) << '\n';
        out << "\"mean-absolute error\": " << meanAbsE << '\n';
        out << "\"relative-squared error\": " << r2E << '\n';
        out << "\"root relative-squared error\": " << sqrt(r2E) << '\n';
        out << "\"relative-mean error\": " << rAbsE << '\n';
        out << "\"r2error\": " << 1-r2E << '\n';
        out << "\"pearson correlation\": " << pearsonCorr << '\n';
        out << "\"OOV docs\": " << getOOVDocs()  << '\n';
        out <<	 "}";
    }
}


void SEQLPredictor::print_class_stats(std::string filename){
    std::streambuf * buf;
    std::ofstream of;

    if(!filename.empty()){
        of.open(filename);
        buf = of.rdbuf();
    } else {
        buf = std::cout.rdbuf();
    }

    std::ostream out(buf);
    double prec = 1.0 * stats.TP/(stats.TP + stats.FP);
    if (stats.TP + stats.FP == 0) prec = 0;
    double rec  = 1.0 * stats.TP/(stats.TP + stats.FN);
    if (stats.TP + stats.FN == 0) rec = 0;
    double f1 =  2 * rec * prec / (prec+rec);
    if (prec + rec == 0) f1 = 0;

    double specificity = 1.0 * stats.TN/(stats.TN + stats.FP);
    if (stats.TN + stats.FP == 0) specificity = 0;
    // sensitivity = recall
    double sensitivity  = 1.0 * stats.TP/(stats.TP + stats.FN);
    if (stats.TP + stats.FN == 0) sensitivity = 0;
    double fss =  2 * specificity * sensitivity / (specificity + sensitivity);
    if (specificity + sensitivity == 0) fss = 0;

    // Sort the scores ascendingly by the predicted score.
    sort(scores.begin(), scores.end());

    double AUC = SEQLStats::calcROC(scores);
    double AUC50 = SEQLStats::calcROC50(scores);
    double balanced_error = 0.5 * ((1.0 * stats.FN / (stats.TP + stats.FN)) + (1.0 * stats.FP / (stats.FP + stats.TN)));
    unsigned int correct = stats.TP + stats.TN;
    unsigned int all = stats.TP + stats.FP + stats.FN + stats.TN;

    std::printf ("Classif Threshold:   %.5f\n", -model->get_bias());
    std::printf ("Accuracy:   %.5f%% (%d/%d)\n", 100.0 * correct / all , correct, all);
    std::printf ("Error:      %.5f%% (%d/%d)\n", 100.0 - 100.0 * correct / all, all - correct, all);
    std::printf ("Balanced Error:     %.5f%%\n", 100.0 * balanced_error);
    std::printf ("AUC:        %.5f%%\n", AUC);
    //std::printf ("(1 - AUC):   %.5f%%\n", 100 - AUC);
    std::printf ("AUC50:      %.5f%%\n", AUC50);
    std::printf ("Precision:  %.5f%% (%d/%d)\n", 100.0 * prec,  stats.TP, stats.TP + stats.FP);
    std::printf ("Recall:     %.5f%% (%d/%d)\n", 100.0 * rec, stats.TP, stats.TP + stats.FN);
    std::printf ("F1:         %.5f%%\n",         100.0 * f1);
    std::printf ("Specificity:  %.5f%% (%d/%d)\n", 100.0 * specificity,  stats.TN, stats.TN + stats.FP);
    std::printf ("Sensitivity:     %.5f%% (%d/%d)\n", 100.0 * sensitivity, stats.TP, stats.TP + stats.FN);
    std::printf ("FSS:         %.5f%%\n",         100.0 * fss);

    std::printf ("System/Answer p/p p/n n/p n/n: %d %d %d %d\n", stats.TP,stats.FP,stats.FN,stats.TN);
    std::printf ("OOV docs:   %d\n", getOOVDocs());

  if(!filename.empty()){
      out <<	 "{";
      out <<		 "\"Classif Threshold\": " <<	 -model->get_bias() <<		  ",\n";
      out <<		 "\"Accuracy\": " <<		 100.0 * correct / all <<  	  ",\n";
      out <<		 "\"Error\": " <<		 100.0 - 100.0 * correct / all << ",\n";
      out <<		 "\"Balanced Error\": " <<	 100.0 * balanced_error <<	  ",\n";
      out <<		 "\"AUC\": " <<			 AUC <<				  ",\n";
      out <<		 "\"Precision\": " <<		 100.0 * prec <<		  ",\n";
      out <<		 "\"Recall\": " <<		 100.0 * rec <<			  ",\n";
      out <<		 "\"F1\": " <<			 100.0 * f1 <<			  ",\n";
      out <<		 "\"Specificity\": " <<		 100.0 * specificity <<		  ",\n";
      out <<		 "\"Sensitivity\": " <<		 100.0 * sensitivity <<		  ",\n";
      out <<		 "\"FSS\": " <<			 100.0 * fss <<		       	  ",\n";
      out <<		 "\"TruePositive\": " <<	 stats.TP <<			  ",\n";
      out <<		 "\"FalsePositive\": " <<	 stats.FP <<			  ",\n";
      out <<		 "\"TrueNegative\": " <<	 stats.TN <<			  ",\n";
      out <<		 "\"FalseNegative\": " <<	 stats.FN <<			  ",\n";
      out <<		 "\"OOV docs\": " <<		 getOOVDocs() <<                  "\n";
      out <<	 "}";
    }
  of.close();
};

void SEQLPredictor::tune(std::string filename){
    struct timeval t;
    struct timeval t_origin;

    gettimeofday(&t_origin, NULL);

    cout << "\nreading training file for classif_tune_threshold...\n\n";
    evalFile(filename);
    sort(scores.begin(), scores.end());
    double AUC = SEQLStats::calcROC(scores);

    std::printf ("AUC:       %.5f%%\n", AUC);
    std::printf ("(1 - AUC): %.5f%%\n", (100 - AUC));
    // Choose the threshold that minimized the errors on training data.
    // Same as Madigan et al BBR.

    // Start by retrieving all, e.g. predict all as positives.
    // Compute the error as FP + FN.
    unsigned int all = stats.TP + stats.FP + stats.FN + stats.TN;
    unsigned int TP = stats.TP + stats.FN;
    unsigned int FP = all - stats.TP + stats.FN;
    unsigned int FN = 0;
    unsigned int TN = 0;
    unsigned int min_error = FP + FN;
    unsigned int current_error = 0;
    double best_threshold = -numeric_limits<double>::max();

    for (unsigned int i = 0; i < all; ++i) {
        // Take only 1st in a string of equal values
        if (i != 0 && scores[i].first > scores[i-1].first) {
            current_error = FP + FN; // sum of errors, e.g # training errors
            if (current_error < min_error) {
                min_error = current_error;
                best_threshold = (scores[i-1].first + scores[i].first) / 2;
                //cout << "\nThreshold: " << best_threshold;
                //cout << "\n# errors (FP + FN): " << min_error;
                //std::printf ("\nAccuracy: %.5f%% (%d/%d)\n", 100.0 * (TP + TN) / all, TP + TN, all);
            }
        }
        if (scores[i].second > 0) {
            FN++; TP--;
        }else{
            FP--; TN++;
        }
    }

    // Finally, check the "retrieve none" situation
    current_error = FP + FN;
    if (current_error < min_error) {
        min_error = current_error;
        best_threshold = scores[all-1].first + 1;
        //cout << "\nThreshold (retrieve none): " << best_threshold;
        //cout << "\n# errors (FP + FN): " << min_error;
        //std::printf ("\nAccuracy: %.5f%% (%d/%d)\n", 100.0 * (TP + TN) / all, TP + TN, all);
    }

    // This procedure finds best_threshold such as if(predicted_score > best_threshold) classify pos;
    // Our seql_classify code uses predicted_score + bias > 0, thus we need to take -threshold.

    gettimeofday(&t, NULL);
    cout << "end classification( " << (t.tv_sec - t_origin.tv_sec) << " seconds; " << (t.tv_sec - t_origin.tv_sec) / 60.0 << " minutes )\n";
    cout.flush();

    //    cout << "\nBest Threshold: " << best_threshold;
    cout << "\n# errors (FP + FN): " << min_error;
    std::printf ("\nAccuracy: %.5f%% (%d/%d)\n", 100.0 * (all - min_error) / all, all - min_error, all);

    //      std::cout << "\nBias (-best_threshold):" << -best_threshold << std::endl;
    cout << "\nBest threshold: " << best_threshold << "\n";
    model->set_bias(-best_threshold);

}

//Compute r-squared score
double SEQLStats::calcR2(const std::vector< std::pair<double, double> >& scores,const double meanY)
{
    double SSTotal = 0;
    double SSRes =0;
    for( auto const& scorePair : scores)
        {
            SSTotal += pow(scorePair.second - meanY, 2);
            SSRes += pow(scorePair.second - scorePair.first, 2);
            // if( verbose >=3){
            //     std::cout << "Real: " <<scorePair.first << " Predict: " << scorePair.second << " Mean: " << meanY << std::endl;
            // }
            // if (verbose >=4){
            //     std::cout << "SSTotal: "<< SSTotal << " SSRes: " << SSRes << " " << std::endl;
            // }
        }
    double r2 = (SSRes/SSTotal);
    std::cout << "SSTotal:" << SSTotal << " SSRes: " << SSRes << std::endl;
    return r2;
}

//Compute r-absolute error
double SEQLStats::calcRabs(const std::vector< std::pair<double, double> >& scores,const double meanY)
{
    double SSTotal = 0;
    double SSRes =0;
    for( auto const& scorePair : scores)
        {
            SSTotal += abs(scorePair.second - meanY);
            SSRes += abs(scorePair.second - scorePair.first);
        }
    double r =(SSRes/SSTotal);

    return r;
}

// Compute the area under the ROC curve.
double SEQLStats::calcROC(const std::vector< std::pair<double, double> >& forROC )
{
    //std::sort( forROC.begin(), forROC.end() );
    double area = 0;
    double x=0, xbreak=0;
    double y=0, ybreak=0;
    double prevscore = - numeric_limits<double>::infinity();
    for(auto ritr=forROC.rbegin(); ritr!=forROC.rend(); ritr++ )
        {
            double score = ritr->first;
            int label = ritr->second;
            //cout << "\nscore: " << score << " label: " << label;
            if( score != prevscore ) {
                //cout << "\nx: " << x << " xbreak: " << xbreak << " y: " << y << " ybreak: " << ybreak;
                area += (x-xbreak)*(y+ybreak)/2.0;
                //cout << "\narea: " << area;
                xbreak = x;
                ybreak = y;
                prevscore = score;
            }
            if( label > 0)  y ++;
            else     x ++;
        }
    area += (x-xbreak)*(y+ybreak)/2.0; //the last bin
    if( 0==y || x==0 )   area = 0.0;   // degenerate case
    else        area = 100.0 * area /( x*y );
    return area;
}

// Compute the area under the ROC50 curve.
// Fixes the number of negatives to 50.
// Stop computing curve after seeing 50 negatives.
double SEQLStats::calcROC50(const std::vector< std::pair<double, double> >& forROC )
{
    //std::sort( forROC.begin(), forROC.end() );
    double area50 = 0;
    double x=0, xbreak=0;
    double y=0, ybreak=0;
    double prevscore = - numeric_limits<double>::infinity();
    for(auto ritr=forROC.rbegin(); ritr!=forROC.rend(); ritr++ )
        {
            double score = ritr->first;
            int label = ritr->second;

            if( score != prevscore && x < 50) {
                area50 += (x-xbreak)*(y+ybreak)/2.0;
                xbreak = x;
                ybreak = y;
                prevscore = score;
            }
            if( label > 0)  y ++;
            else if (x < 50) x ++;
        }
    area50 += (x-xbreak)*(y+ybreak)/2.0; //the last bin
    if( 0==y || x==0 )   area50 = 0.0;   // degenerate case
    else        area50 = 100.0 * area50 /( 50*y );
    return area50;
}

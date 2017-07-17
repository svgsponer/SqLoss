/*
 * Author: Severin Gsponer (severin.gsponer@insight-centre.org)
 * File that runs SEQL with squared error loss on the Microsoft malware classification challange.
 *
 */

#include "../seql_learn.h"
#include "../seql_predictor.h"
#include "../FileReader.cpp"


void prepareInvertedIndex (std::map<string, SNode>& seed, std::vector<std::string>& transaction) {
    string unigram;
    bool at_space = false;

    // Prepare the locations for unigrams.
    cout << "\nprepare inverted index for unigrams";

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

int run(const int pos_class,
        const std::vector<std::string>& x,
        const std::vector<double>& y,
        const std::vector<std::string>& xval,
        const std::vector<double>& yval,
        std::map<string, SNode>& seed,
        const int foldnr){

    std::cout << "Start posClass " << pos_class << std::endl;

    // Vector to store modified (binary) y's for certain class
    std::vector<double> y_class{};
    y_class.reserve(y.size());
    std::transform(y.begin(), y.end(), std::back_inserter(y_class),
                   [=](int lab) { return lab==pos_class?1:-1; } );


    // Vector to store modified (binary) y's for certain class
    std::vector<double> yval_class{};
    yval_class.reserve(yval.size());
    std::transform(yval.begin(), yval.end(), std::back_inserter(yval_class),
                   [=](int lab) { return lab==pos_class?1:-1; } );

    // Setup all filenames
    std::string basename = "MMC_" +std::to_string(foldnr) + "_" + std::to_string(pos_class);
    std::string modelCreationFile{basename + ".modelCreation"};
    std::string modelBinFile{basename + ".bin"};
    std::string modelFile{basename + ".model"};
    std::string predictionFile{basename + ".conc.pred"};
    std::string statsFile{basename + ".stats.json"};
    std::string csvFile{basename + ".itrStats.csv"};


    SeqLearner seql_learner {x,y_class,
                             3, // objective
                             20, //maxpat
                             1,        //minpat
                             0,        //minsup
                             0,        //maxgap 
                             0,        //maxcongap
                             1,        //token_type
                             0,        //traversal_strategy
                             0.005,    //convergence_threshold
                             5,        //regularizer_weight
                             1,      //alpha
                             2,        //verbostity
                             true};      //csvLog
    auto start = std::chrono::steady_clock::now();
    LinearModel::LinearModel model =  seql_learner.learn (5000, modelCreationFile, 0,
                                                          xval, yval_class, csvFile, seed);
    auto end = std::chrono::steady_clock::now();

    cout << "\nTotal time taken to run learner: "
         << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
         << "ms"<< endl;

    model.print_fulllist(modelCreationFile);
    model.build_tree(1);
    model.set_bias(0);
    model.save_as_binary(modelBinFile);
    model.print_model(modelFile);

    // Evaluation
    SEQLPredictor predictor{1, 1, &model};
    std::string testFile{"INSERT TEST FILE PATH" + std::to_string(foldnr)};
    predictor.evalFile(testFile, predictionFile, pos_class);
    predictor.print_class_stats(statsFile);
    std::cout << "end class " << pos_class << std::endl;
    return 0;
}

int main (int argc, char **argv)
{
    int foldnr = 0;
    if (argc != 2){
        std::cout << "ERROR: You have to provide the number of the fold you want to calculate" << std::endl;
        std::exit(-1);
    }else{
        foldnr = std::stoi(argv[1]);
    }
    int number_of_classes = 9;

    // Read the input file
    std::vector<std::string> x;
    std::vector<double> y;
    std::string inputFile{"INSERT TRAINING FILE" + std::to_string(foldnr)};
    std::tie(x,y, std::ignore) = SEQL::read_input(inputFile);

    // Read validation file
    std::vector<std::string> xval;
    std::vector<double> yval;
    std::string validationFile{"INSET VALIDATION FILE (/dev/null if none)" + std::to_string(foldnr)};
    std::tie(xval,yval, std::ignore) = SEQL::read_input(validationFile);

    std::map<string, SNode> seed;
    prepareInvertedIndex(seed, x);

    for(int pos_class=1; pos_class <= number_of_classes; pos_class++){
        run(pos_class, x, y, xval, yval, seed, foldnr);
    }

    return 0;
};


/*
 * Author: Severin Gsponer (severin.gsponer@insight-centre.org)
 * Programm to start the regression pipeline
 * At least provide training file, testfile and a basename for all the output files. 
 */

#include "../seql_learn.h"
#include "../seql_predictor.h"
#include "../FileReader.cpp"

#define OPT "[-t theshold] [-o objective_function] [-O csvFile] [-m minsup] [-l minpat] [-L maxpat] [-g maxgap] [-G maxcongap] [-r traversal_strategy ] [-T #round] [-n token_type] [-c convergence_threshold] [-C regularizer_value] [-a l1_vs_l2_regularizer_weight] [-v verbosity] training_file test_file basename validationset_file"


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

int main (int argc, char **argv)
{
    unsigned int objective = 3;
    // Word or character token type. By default char token.
    bool token_type = 1;

    // Pattern properties
    unsigned int maxpat = 0xffffffff;
    unsigned int minpat = 1;
    unsigned int maxitr = 5000;
    unsigned int minsup = 1;
    // Max # of total wildcards allowed in a feature.
    unsigned int maxgap = 0;
    // Max # of consec wildcards allowed in a feature.
    unsigned int maxcongap = 0;

    // BFS vs DFS traversal. By default BFS.
    bool traversal_strategy = 0;

    // The C regularizer parameter in regularized loss formulation. It constraints the weights of features.
    // C = 0 no constraints (standard SLR), the larger the C, the more the weights are shrinked towards each other (using L2) or towards 0 (using L1)
    double C = 1;
    // The alpha parameter decides weight on L1 vs L2 regularizer: alpha * L1 + (1 - alpha) * L2. By default we use an L2 regularizer.
    double alpha = 0.2;

    double convergence_threshold = 0.005;
    int verbosity = 1;

    std::string csvFile;
    bool csvLog = false;

    double threshold = 0;
    double mean = 0;
    int opt;
    while ((opt = getopt(argc, argv, "t:o:T:L:l:m:g:G:n:r:c:C:a:v:O:")) != -1) {
        switch(opt) {
        case 'o':
            objective = atoi (optarg);
            break;
        case 'T':
            maxitr = atoi (optarg);
            break;
        case 'L':
            maxpat = atoi (optarg);
            break;
        case 'l':
            minpat = atoi (optarg);
            break;
        case 'm':
            minsup = atoi (optarg);
            break;
        case 'g':
            maxgap = atoi (optarg);
            break;
        case 'G':
            maxcongap = atoi (optarg);
            break;
        case 'n':
            token_type = atoi (optarg);
            break;
        case 'r':
            traversal_strategy = atoi (optarg);
            break;
        case 'c':
            convergence_threshold = atof (optarg);
            break;
        case 'C':
            C = atof (optarg);
            break;
        case 'a':
            alpha = atof (optarg);
            break;
        case 'v':
            verbosity = atoi (optarg);
            break;
        case 't':
            threshold = atof(optarg);
            break;
        case 'O':
            csvFile = optarg;
            csvLog = true;
            break;
        default:
            std::cout << "Usage: " << argv[0] << OPT << std::endl;
            return -1;
        }
    }

    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << OPT << std::endl;
        return -1;
    }
    std::string inputFile{argv[argc-4]};
    std::string validationSet{argv[argc-1]};
    std::string basename = {argv[argc-2]};
    // const char* modelCreationFile{basename + ".model"};
    std::string modelCreationFile{basename + ".modelCreation"};
    std::string modelBinFile{basename + ".bin"};
    std::string modelFile{basename + ".model"};
    std::string predictionFile{basename + ".conc.pred"};
    std::string statsFile{basename + ".stats.json"};
    std::string testFile{argv[argc-3]};

    std::vector<std::string> x;
    std::vector<double> y;
    std::tie(x,y, std::ignore) = SEQL::read_input(inputFile);

    std::vector<std::string> xval;
    std::vector<double> yval;
    std::tie(xval,yval, std::ignore) = SEQL::read_input(validationSet);
    std::map<string, SNode> seed;
    prepareInvertedIndex(seed, x);


    SeqLearner seql_learner {x,y,
                            objective,
                            maxpat,
                            minpat,
                            minsup,
                            maxgap,
                            maxcongap,
                            token_type,
                            traversal_strategy,
                            convergence_threshold,
                            C,
                            alpha,
                            verbosity,
                            csvLog};
    auto start = std::chrono::steady_clock::now();
    LinearModel::LinearModel model =  seql_learner.learn (maxitr, modelCreationFile, csvFile, seed);
    auto end = std::chrono::steady_clock::now();
    cout << "\nTotal time taken to run learner: "
         << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
         << "ms"<< endl;
    model.print_fulllist(modelCreationFile);
    model.build_tree(1);
    model.set_bias(threshold);
    model.save_as_binary(modelBinFile);
    model.print_model(modelFile);

    SEQLPredictor predictor{1, token_type, &model};
    // // classifier.open(modelBinFile, 0);
    predictor.evalFile(testFile, predictionFile );
    predictor.print_class_stats(statsFile);
    return 0;
}

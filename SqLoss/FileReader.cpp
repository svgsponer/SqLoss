#include "memory"
#include <cstring>
#include "common.h"

namespace SEQL{
    using namespace std;
    // Limit allows to limit how many lines will be read
    std::tuple<vector<std::string>, vector<double>, double> read_input (std::string filename, int limit = -1) {
        // Set the max line/document size to (10Mb).
        cout<<"Read file: " << filename << std::endl;
        constexpr int kMaxLineSize = 10000000;
        char* line {new char[kMaxLineSize]};
        char* column[5];
        unsigned int num_pos = 0;
        unsigned int num_neg = 0;
        double sum = 0;
        string doc;
        std::vector<double> y;
        std::vector<std::string> x;

        std::unique_ptr<std::istream> ifs {nullptr};
        if (filename == "-") ifs.reset(&std::cin);
        else              ifs.reset(new std::ifstream(filename.c_str()));

        if (! *ifs){
            std::cerr << "Error: " << filename << " No such file or directory" << std::endl;
            std::exit (1);
        }
        cout << "read() input data....";

        while (ifs->getline (line, kMaxLineSize) && limit !=0) {
            if (line[0] == '\0' || line[0] == ';') continue;
            if (line[strlen(line) - 1 ] == '\r')
                line[strlen(line) - 1 ] = '\0';

            if (2 != tokenize (line, "\t ", column, 2)) {
                std::cerr << "FATAL: Format Error: " << line << std::endl;
                std::exit(1);
            }

            // Prepare doc. Assumes column[1] is the original text, e.g. no bracketing of original doc.
            doc.assign(column[1]);
            if(doc.empty()){
                continue;
            };
            x.push_back(doc);

            // Prepare class. _y is +1/-1 for classification or a double in case of regression.
            double _y = atof (column[0]);
            y.push_back (_y);

            if (_y > 0) num_pos++;
            else num_neg++;
            sum += _y;

            cout.flush();

            limit--;
        }
        double mean= sum/(num_pos + num_neg);
        cout << "\n# positive samples: " << num_pos;
        cout << "\n# negative samples: " << num_neg;
        cout << "\nTotal points: " << (num_pos + num_neg);
        cout << "\nMean of y values: " << mean;
        cout << "\nend read() input data....\n\n";

        delete [] line;

        return make_tuple(x, y, mean);
    }

}

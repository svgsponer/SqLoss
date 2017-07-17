/** Linear model class
    Author: Severin Gsponer (severin.gsponer@insight-centre.com)
**/
#include "LinearModel.h"

namespace LinearModel{
    template <typename T1, typename T2>
    struct pair_2nd_cmp: public std::binary_function<bool, T1, T2> {
        bool operator () (const std::pair <T1, T2>& x1, const std::pair<T1, T2> &x2)
        {
            return x1.second > x2.second;
        }
    };

    void LinearModel::print_fulllist(std::string outfile_name){
        std::ofstream os (outfile_name, std::ofstream::out);
        if (! os) {
            std::cerr << "FATAL: Cannot open output file: " << outfile_name << std::endl;
        }

        os.setf(std::ios::fixed,std::ios::floatfield);
        os.precision(17);

        for(auto itr : fullList){
            os<< std::get<0>(itr) << ' ' << std::get<1>(itr) << "\n";
        }
        os.flush();
    };

    // Saves the doubleArray and vector of weights in a binary file
    void LinearModel::save_as_binary(std::string outfile_name){
        std::ofstream ofs (outfile_name, std::ios::binary|std::ios::out);

        if (!ofs) {
            std::cerr << "FATAL: Cannot open outputfile: " << outfile_name << std::endl;
            exit(1);
        }

        // Use save funciton of double array first since it corups data in the offset section
        da->save(outfile_name.c_str(), "wb", sizeof(size_t) + sizeof(std::vector<double>::value_type) * weights.size());
        size_t s = weights.size();
        ofs.write (reinterpret_cast<const char *>(&s), sizeof(size_t));
        ofs.write (reinterpret_cast<const char *>(&weights[0]), sizeof (double) * weights.size());
        ofs.close ();
    }

    void LinearModel::print_model(std::string outfile_name){
        std::ofstream ofs (outfile_name, std::ios::out);

        if (!ofs) {
            std::cerr << "FATAL: Cannot open outputfile: " << outfile_name << std::endl;
            exit(1);
        }
        ofs.precision (24);
        ofs << bias << std::endl;
        ofs << intercept << " INTERCEPT" << std::endl;
        std::sort (begin(rules), end(rules), pair_2nd_cmp <const char*, double>());
        for (auto rule :rules){
            ofs << rule.second << " " << rule.first << std::endl;
        }
    };

    void LinearModel::add(double step_length, std::string ngram){
        fullList.push_back(std::make_tuple(step_length, ngram));

    };

    // Each feature weight will be multiplied by MultiplyValue. For classification set it to 2
    // Old classifier normalized the weight this will be now done in an additional step by call normalizeWeights
    void LinearModel::build_tree(double multiplyValue){

        std::unique_ptr<Darts::DoubleArray> newDa(new Darts::DoubleArray);
        // Darts::DoubleArray newDa;

        std::vector <Darts::DoubleArray::key_type *> ary;
        std::map<std::string, double> rulesMap;
        weights.clear();
        rules.clear();
        double alpha_sum = 0.0;
        for(auto element:fullList ){
            rulesMap[std::get<1>(element)] += multiplyValue * std::get<0>(element);
            alpha_sum += std::abs(std::get<0>(element));
            bias -= std::get<0>(element);
        }

        bias /= alpha_sum;
        //bias = 0;
        l1_norm = alpha_sum;

        intercept = rulesMap["*INTERCEPT*"];
        rulesMap.erase("*INTERCEPT*");
        for (auto rule : rulesMap) {

            // double a = rule.second / alpha_sum;
            double a = rule.second ;
            l2_norm +=  pow(rule.second, 2);

            rules.push_back (std::make_pair (rule.first.c_str(), a));
            ary.push_back  ((Darts::DoubleArray::key_type *)rule.first.c_str());
            weights.push_back (a);
        }

        l2_norm = pow(l2_norm, 0.5);

        std::cout << "Total: " << weights.size() << " rule(s)" << std::endl;
        std::cout << "l1_norm: " << l1_norm << ", l2_norm: " << l2_norm << std::endl;

        if (ary.empty()) {
            std::cerr << "FATAL: no features in the model" << std::endl;
            exit(1);
        }
        if (newDa->build (ary.size(), &ary[0], nullptr, nullptr, nullptr) != 0) {
            std::cerr << "Error: cannot build double array  " << std::endl;
            exit(1);
        }
        // return newDa;
        da = std::move(newDa);
    };

    void LinearModel::normalize_weights(){
        for(auto& elm : weights) {
            elm = elm/l1_norm;
        };
        for(auto& rule : rules) {
            rule.second = rule.second/l1_norm;
        };
    };

    double LinearModel::get_bias() const {
        return bias;
    };

    void LinearModel::set_bias(double bias_){
        bias = bias_;
    };

    bool operator== (const LinearModel &c1, const LinearModel &c2){
        return (c1.fullList == c2.fullList);
    };

    bool LinearModel::open (std::string file, double threshold)
    {
        std::ifstream ifs (file, std::ios::binary|std::ios::out);

        if (!ifs) {
            std::cerr << "FATAL: Cannot open outputfile: " << file << std::endl;
            exit(1);
        }
        size_t s = 0;
        ifs.read(reinterpret_cast<char *>(&s), sizeof(size_t));
        weights.resize(s);
        weights.clear();
        ifs.read (reinterpret_cast<char *>(&weights[0]), sizeof (double) * s);
        ifs.close ();

        da->open(file.c_str(), "rb", sizeof(size_t)+sizeof(double)*s );
        bias = -threshold;  //set bias to minus user-provided-thereshold
        return true;
    }
    std::vector<std::tuple<double, std::string>> parse_model(std::string file){
        std::unique_ptr<std::istream> is {nullptr};
        if (file == "-") is.reset(&std::cin);
        else              is.reset(new std::ifstream(file.c_str()));

        if (! *is) {
            std::cerr << "FATAL: Cannot open inputfile: " << file << std::endl;
            exit (1);
        }

        std::vector<std::tuple<double, std::string>> fullList;
        std::vector<std::string> x {};
        std::vector<double> beta {};
        char buf[8192];
        char *column[2];
        std::string doc {};
        while (is->getline (buf, 8192)) {
            if (buf[strlen(buf) - 1] == '\r') {
                buf[strlen(buf) - 1] = '\0';
            }

            if (2 != tokenize (buf, "\t ", column, 2)) {
                std::cerr << "FATAL: Format Error: " << buf << std::endl;
                exit(1);
            }
            // std::cout << "\n" << column[0] << " " << column[1] ;
            // std::cout.flush();
            // Ignore rules containing only 1 character.
            //if (strlen(column[1]) <= 1) continue;
            double curbeta =  atof (column[0]);
            doc.assign(column[1]);
            fullList.push_back(make_tuple(curbeta, doc));

        }

        return fullList;
    };
};


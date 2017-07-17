/** Linear model class
    Author: Severin Gsponer (severin.gsponer@insight-centre.com)
    The model consist of is represented as vector of tupels from the step_size and the corresponding ngram
**/

#ifndef LINEARMODEL_H
#define LINEARMODEL_H

#include <vector>
#include <tuple>
#include <iostream>
#include <fstream>
#include <string>
#include <map>
#include "unistd.h"
#include <cstring>
#include "common.h"
#include "darts.h"
#include <memory>
#include "mmap.h"

namespace LinearModel{
class LinearModel{
 public:
    LinearModel():da(new Darts::DoubleArray){};
    LinearModel(std::vector<std::tuple<double, std::string>> fullList):fullList(fullList){};
    LinearModel(std::string filename, double threshold):da(new Darts::DoubleArray){
        open(filename, threshold);
    };

    void add(double step_length, std::string ngram);
    void print_fulllist(std::string outfile_name);
    void print_model(std::string outfile_name);
    void save_as_binary(std::string outfile_name);
    void build_tree(double multiplyValue = 1);
    friend bool operator== (const LinearModel &c1, const LinearModel &c2);
    std::shared_ptr<Darts::DoubleArray> da;

    bool open (std::string file, double threshold);
    double get_bias() const;
    void set_bias(double bias);
    std::vector <double> weights;
    void normalize_weights();
    long double intercept = 0.0;
 private:
    double bias = 0.0;
    double l1_norm = 0.0;
    double l2_norm = 0.0;
    std::vector <std::pair<const char *, double> > rules;
    std::vector<std::tuple<double, std::string>> fullList;

};

    std::vector<std::tuple<double, std::string>> parse_model(std::string file);
};

#endif

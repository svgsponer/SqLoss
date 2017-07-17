#include <vector>
#include <string>
#include <iostream>
#include "common_string_symbol.h"

void str2node (const char *str, std::vector <stx::string_symbol>& doc, int token_type)
{
    unsigned int len = strlen (str);
    bool at_space = false;
    std::string unigram = "";

    for (unsigned int pos = 0; pos < len; ++pos) {
        // Skip white spaces. They are not considered as unigrams.
        if (isspace(str[pos])) {
            at_space = true;
            continue;
        }
        // If word level tokens.
        if (!token_type) {
            if (at_space || pos == 0) {
                at_space = false;

                if (!unigram.empty()) {
                    doc.push_back(unigram);
                    unigram.clear();
                }
                unigram += str[pos];
            } else {
                unigram += str[pos];
            }
        } else {
            // Char (i.e. byte) level token.
            unigram = str[pos];
            doc.push_back(unigram);
            unigram.clear();
        }
    }

    if (!token_type) {
        if (!unigram.empty()) {
            doc.push_back(unigram);
            unigram.clear();
        }
    }
}

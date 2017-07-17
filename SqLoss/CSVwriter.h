/* /\** */
/*  *   \file CSVwriter.h */
/*  *   \brief Class for to write a CSV file */
/*  * */
/*  * */
/*  * */
/*  *\/ */
#ifndef CSVWRITER_H
#define CSVWRITER_H

#include <iostream>

class CSVwriter{
 private:
    std::string filename;
    std::ofstream outfile;
    void open(){
        outfile.open(filename, std::ios::trunc );
        if(!outfile.is_open()){
            std::cout << "ERROR WITH CSVFILE FILE";
        }
    }
 public:
    /**
     * A specialization to stream the last argument
     * and terminate the recursion.
     */
    template<typename Arg1>
        void DoLog( const Arg1 & arg1)
    {
        outfile << arg1 << std::endl;
    }

    /**
     * Recursive function to keep streaming the arguments
     * one at a time until the last argument is reached and
     * the specialization above is called.
     */
    template<typename Arg1, typename... Args>
        void DoLog(const Arg1 & arg1, const Args&... args)
    {
        outfile << arg1 << ",";
        DoLog(args...);
    }

 CSVwriter(std::string fname = "csvwriter.csv" ): filename(fname) {open();};
    ~CSVwriter(){outfile.close();};
};
#endif

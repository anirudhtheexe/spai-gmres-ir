#include "bspai/cpu.hpp"

#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: cpu_matrix_probe MATRIX.mtx\n";
        return 2;
    }
    try {
        std::ifstream input(argv[1]);
        if (!input) throw std::runtime_error("Cannot open matrix file");
        const bspai::CsrMatrix matrix = bspai::read_matrix_market(input);
        const std::vector<double> product = matrix.multiply(std::vector<double>(matrix.cols(), 1.0));
        std::cout << "rows " << matrix.rows() << " cols " << matrix.cols()
                  << " nnz " << matrix.values().size() << '\n' << std::setprecision(17);
        for (std::size_t row = 0; row < matrix.rows(); ++row) {
            for (std::size_t index = matrix.row_offsets()[row]; index < matrix.row_offsets()[row + 1]; ++index)
                std::cout << "entry " << row << ' ' << matrix.column_indices()[index] << ' ' << matrix.values()[index] << '\n';
            std::cout << "A_times_ones " << row << ' ' << product[row] << '\n';
        }
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}

#ifndef BSPAI_CPU_HPP
#define BSPAI_CPU_HPP

#include <cstddef>
#include <istream>
#include <vector>

namespace bspai {

struct Entry {
    std::size_t row;
    std::size_t col;
    double value;
};

// Canonical CSR: sorted column indices, summed duplicates, no stored zeros.
class CsrMatrix {
  public:
    static CsrMatrix from_entries(std::size_t rows, std::size_t cols, std::vector<Entry> entries);
    std::size_t rows() const {
        return rows_;
    }
    std::size_t cols() const {
        return cols_;
    }
    const std::vector<std::size_t> &row_offsets() const {
        return offsets_;
    }
    const std::vector<std::size_t> &column_indices() const {
        return columns_;
    }
    const std::vector<double> &values() const {
        return values_;
    }
    std::vector<double> multiply(const std::vector<double> &x) const;
    CsrMatrix transpose() const;

  private:
    CsrMatrix(std::size_t rows, std::size_t cols);
    std::size_t rows_, cols_;
    std::vector<std::size_t> offsets_, columns_;
    std::vector<double> values_;
};

// Matrix Market coordinate real/integer, general/symmetric only.
// Rejects nonfinite values, unsupported formats and malformed input.
CsrMatrix read_matrix_market(std::istream &input);

} // namespace bspai
#endif

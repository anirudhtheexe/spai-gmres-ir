#include "bspai/cpu.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>

namespace bspai {
namespace {

bool next_data_line(std::istream& input, std::string& line) {
    while (std::getline(input, line)) {
        const std::size_t begin = line.find_first_not_of(" \t\r");
        if (begin != std::string::npos && line[begin] != '%') return true;
    }
    return false;
}

std::size_t parse_size(const std::string& token) {
    if (token.empty() || token.find_first_not_of("0123456789") != std::string::npos)
        throw std::runtime_error("Expected a nonnegative integer dimension/index");
    std::size_t result = 0;
    for (char digit : token) {
        const std::size_t number = static_cast<std::size_t>(digit - '0');
        if (result > (std::numeric_limits<std::size_t>::max() - number) / 10)
            throw std::runtime_error("Dimension/index exceeds size_t");
        result = result * 10 + number;
    }
    return result;
}

void require_end(std::istringstream& line) {
    std::string extra;
    if (line >> extra) throw std::runtime_error("Extra field in Matrix Market line");
}

} // namespace

CsrMatrix::CsrMatrix(std::size_t rows, std::size_t cols)
    : rows_(rows), cols_(cols) {
    if (rows == std::numeric_limits<std::size_t>::max())
        throw std::invalid_argument("Row count overflows CSR offsets");
    offsets_.assign(rows + 1, 0);
}

CsrMatrix CsrMatrix::from_entries(std::size_t rows, std::size_t cols,
                                  std::vector<Entry> entries) {
    for (const Entry& entry : entries) {
        if (entry.row >= rows || entry.col >= cols)
            throw std::invalid_argument("Entry index outside matrix dimensions");
        if (!std::isfinite(entry.value))
            throw std::invalid_argument("Matrix entries must be finite");
    }
    // Stable order retains the input accumulation order for duplicate entries.
    std::stable_sort(entries.begin(), entries.end(), [](const Entry& a, const Entry& b) {
        return a.row < b.row || (a.row == b.row && a.col < b.col);
    });
    CsrMatrix matrix(rows, cols);
    for (std::size_t i = 0; i < entries.size();) {
        const Entry first = entries[i++];
        double sum = first.value;
        while (i < entries.size() && entries[i].row == first.row && entries[i].col == first.col) {
            sum += entries[i++].value;
            if (!std::isfinite(sum)) throw std::invalid_argument("Duplicate sum is nonfinite");
        }
        if (sum != 0.0) {
            matrix.columns_.push_back(first.col);
            matrix.values_.push_back(sum);
            ++matrix.offsets_[first.row + 1];
        }
    }
    for (std::size_t row = 0; row < rows; ++row)
        matrix.offsets_[row + 1] += matrix.offsets_[row];
    return matrix;
}

std::vector<double> CsrMatrix::multiply(const std::vector<double>& x) const {
    if (x.size() != cols_) throw std::invalid_argument("Vector dimension mismatch");
    for (double value : x)
        if (!std::isfinite(value)) throw std::invalid_argument("Vector must be finite");
    std::vector<double> result(rows_, 0.0);
    for (std::size_t row = 0; row < rows_; ++row) {
        for (std::size_t index = offsets_[row]; index < offsets_[row + 1]; ++index)
            result[row] += values_[index] * x[columns_[index]];
        if (!std::isfinite(result[row])) throw std::overflow_error("Matrix-vector result is nonfinite");
    }
    return result;
}

CsrMatrix CsrMatrix::transpose() const {
    std::vector<Entry> entries;
    entries.reserve(values_.size());
    for (std::size_t row = 0; row < rows_; ++row)
        for (std::size_t index = offsets_[row]; index < offsets_[row + 1]; ++index)
            entries.push_back(Entry{columns_[index], row, values_[index]});
    return from_entries(cols_, rows_, entries);
}

CsrMatrix read_matrix_market(std::istream& input) {
    std::string text, magic, object, format, field, symmetry;
    if (!std::getline(input, text)) throw std::runtime_error("Missing Matrix Market banner");
    std::istringstream banner(text);
    if (!(banner >> magic >> object >> format >> field >> symmetry))
        throw std::runtime_error("Malformed Matrix Market banner");
    require_end(banner);
    if (magic != "%%MatrixMarket" || object != "matrix" || format != "coordinate" ||
        (field != "real" && field != "integer") || (symmetry != "general" && symmetry != "symmetric"))
        throw std::runtime_error("Unsupported Matrix Market format");
    if (!next_data_line(input, text)) throw std::runtime_error("Missing matrix dimensions");
    std::string row_token, col_token, count_token;
    std::istringstream dimensions(text);
    if (!(dimensions >> row_token >> col_token >> count_token))
        throw std::runtime_error("Malformed matrix dimensions");
    require_end(dimensions);
    const std::size_t rows = parse_size(row_token), cols = parse_size(col_token);
    const std::size_t count = parse_size(count_token);
    if (rows == 0 || cols == 0) throw std::runtime_error("Matrix dimensions must be positive");
    if (symmetry == "symmetric" && rows != cols)
        throw std::runtime_error("Symmetric matrix must be square");
    std::vector<Entry> entries;
    for (std::size_t i = 0; i < count; ++i) {
        if (!next_data_line(input, text)) throw std::runtime_error("Truncated coordinate data");
        std::istringstream coordinate(text);
        std::string value_token;
        if (!(coordinate >> row_token >> col_token >> value_token))
            throw std::runtime_error("Malformed coordinate entry");
        require_end(coordinate);
        const std::size_t row = parse_size(row_token), col = parse_size(col_token);
        if (row == 0 || col == 0 || row > rows || col > cols)
            throw std::runtime_error("Matrix Market index outside one-based dimensions");
        std::size_t used = 0;
        double value;
        try { value = std::stod(value_token, &used); }
        catch (const std::exception&) { throw std::runtime_error("Invalid coordinate value"); }
        if (used != value_token.size() || !std::isfinite(value))
            throw std::runtime_error("Coordinate value must be finite and numeric");
        if (field == "integer" && value_token.find_first_not_of("+-0123456789") != std::string::npos)
            throw std::runtime_error("Integer field requires integer value tokens");
        entries.push_back(Entry{row - 1, col - 1, value});
        if (symmetry == "symmetric" && row != col)
            entries.push_back(Entry{col - 1, row - 1, value});
    }
    if (next_data_line(input, text)) throw std::runtime_error("Trailing coordinate data");
    return CsrMatrix::from_entries(rows, cols, entries);
}

} // namespace bspai

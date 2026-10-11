#include "bspai/cpu.hpp"
#include <cmath>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
int failures = 0;
int checks = 0;
void check(bool ok, const std::string& name) {
    ++checks;
    if (!ok) { ++failures; std::cerr << "FAIL: " << name << '\n'; }
}
bspai::CsrMatrix load(const std::string& text) {
    std::istringstream input(text);
    return bspai::read_matrix_market(input);
}
template <class F> void rejects(F f, const std::string& name) {
    bool threw = false;
    try { f(); } catch (const std::exception&) { threw = true; }
    check(threw, name);
}
void equal_values(const std::vector<double>& actual, const std::vector<double>& expected,
                  const std::string& name) {
    check(actual == expected, name);
}
}

int main() {
    using bspai::CsrMatrix;
    using bspai::Entry;
    try {
        // Exact integer arithmetic here deliberately avoids a tolerance that could mask errors.
        const CsrMatrix a = load(
            "%%MatrixMarket matrix coordinate real general\n% tiny nonsymmetric fixture\n"
            "3 3 7\n3 3 3\n1 2 1\n2 1 2\n1 1 4\n2 2 5\n3 2 1\n2 3 1\n");
        check(a.rows() == 3 && a.cols() == 3, "loader dimensions");
        check(a.row_offsets() == std::vector<std::size_t>({0,2,5,7}), "CSR row offsets");
        check(a.column_indices() == std::vector<std::size_t>({0,1,0,1,2,1,2}), "CSR sorted indices");
        equal_values(a.values(), {4,1,2,5,1,1,3}, "CSR values");
        equal_values(a.multiply({1,-2,3}), {2,-5,7}, "nonsymmetric SpMV manufactured solution");
        equal_values(a.multiply({0,0,0}), {0,0,0}, "zero-vector SpMV");
        equal_values(a.transpose().multiply({1,2,3}), {8,14,11}, "transpose orientation");
        check(a.transpose().transpose().row_offsets() == a.row_offsets() &&
              a.transpose().transpose().column_indices() == a.column_indices() &&
              a.transpose().transpose().values() == a.values(), "double transpose canonical identity");
        rejects([&]() { a.multiply({1,2}); }, "SpMV wrong input length");
        const CsrMatrix dup = load(
            "%%MatrixMarket matrix coordinate integer general\n3 4 5\n"
            "1 2 3\n3 4 5\n1 2 -3\n3 4 2\n2 1 0\n");
        check(dup.rows() == 3 && dup.cols() == 4, "rectangular integer matrix");
        check(dup.row_offsets() == std::vector<std::size_t>({0,0,0,1}), "empty rows after duplicate cancellation");
        check(dup.column_indices() == std::vector<std::size_t>({3}), "duplicates coalesced and zeros dropped");
        equal_values(dup.values(), {7}, "duplicate sum");
        equal_values(dup.multiply({1,2,3,4}), {0,0,28}, "rectangular SpMV");
        equal_values(dup.transpose().multiply({1,2,3}), {0,0,0,21}, "rectangular transpose SpMV");
        const CsrMatrix sym = load(
            "%%MatrixMarket matrix coordinate real symmetric\n3 3 4\n"
            "1 1 2\n2 1 3\n2 2 4\n3 2 5\n");
        equal_values(sym.multiply({1,2,3}), {8,26,10}, "symmetric expansion without doubling diagonal");
        check(sym.values().size() == 6, "symmetric expanded nnz");
        const CsrMatrix empty = load("%%MatrixMarket matrix coordinate real general\n2 3 0\n");
        check(empty.row_offsets() == std::vector<std::size_t>({0,0,0}), "zero-nnz row offsets");
        equal_values(empty.multiply({1,2,3}), {0,0}, "zero-nnz multiply");
        const std::string head = "%%MatrixMarket matrix coordinate real general\n";
        rejects([&]() { load("not a matrix market file\n"); }, "invalid banner");
        rejects([&]() { load("%%MatrixMarket matrix array real general\n2 2\n1\n"); }, "unsupported array storage");
        rejects([&]() { load("%%MatrixMarket matrix coordinate complex general\n1 1 1\n1 1 2 3\n"); }, "unsupported complex field");
        rejects([&]() { load("%%MatrixMarket matrix coordinate pattern general\n1 1 1\n1 1\n"); }, "unsupported pattern field");
        rejects([&]() { load("%%MatrixMarket matrix coordinate real skew-symmetric\n2 2 1\n2 1 1\n"); }, "unsupported symmetry");
        rejects([&]() { load("%%MatrixMarket matrix coordinate real symmetric\n2 3 0\n"); }, "nonsquare symmetric matrix");
        rejects([&]() { load(head + "2 2 1\n0 1 1\n"); }, "zero one-based index");
        rejects([&]() { load(head + "2 2 1\n3 1 1\n"); }, "out-of-range row");
        rejects([&]() { load(head + "2 2 1\n1 3 1\n"); }, "out-of-range column");
        rejects([&]() { load(head + "2 2 2\n1 1 1\n"); }, "truncated coordinate count");
        rejects([&]() { load(head + "2 2 1\n1 1 1\n2 2 2\n"); }, "trailing unexpected coordinate");
        rejects([&]() { load(head + "2 2 1\n1 1 NaN\n"); }, "nonfinite input");
        rejects([&]() { load(head + "2 2 1\n1 1 1e999\n"); }, "overflowing input value");
        rejects([&]() { load(head + "2 2 -1\n"); }, "negative entry count");
        rejects([&]() { CsrMatrix::from_entries(2,2, {Entry{2,0,1}}); }, "direct CSR invalid index");
        rejects([&]() { CsrMatrix::from_entries(2,2, {Entry{0,0,std::numeric_limits<double>::infinity()}}); }, "direct CSR nonfinite value");
        rejects([&]() { CsrMatrix::from_entries(2,2, {Entry{0,0,std::numeric_limits<double>::max()}, Entry{0,0,std::numeric_limits<double>::max()}}); }, "duplicate sum overflow");
        const CsrMatrix rounding = CsrMatrix::from_entries(1,3, {Entry{0,0,0.1},Entry{0,1,0.2},Entry{0,2,0.3}});
        const double actual = rounding.multiply({1,1,1})[0];
        const long double expected = static_cast<long double>(0.1) + static_cast<long double>(0.2) + static_cast<long double>(0.3);
        const long double bound = 8 * std::numeric_limits<double>::epsilon() * (std::fabs(expected));
        check(std::fabs(static_cast<long double>(actual)-expected) <= bound, "floating SpMV independent wider accumulation bound");
    } catch (const std::exception& e) {
        ++failures;
        std::cerr << "UNEXPECTED EXCEPTION: " << e.what() << '\n';
    }
    std::cout << "Independent CPU validation: " << checks << " checks, " << failures << " failures\n";
    std::cout << "double digits=" << std::numeric_limits<double>::digits
              << ", long double digits=" << std::numeric_limits<long double>::digits << '\n';
    return failures ? 1 : 0;
}

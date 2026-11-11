#include "../dfhelper_jk_blocking.h"

#include <iostream>
#include <string>

using Blocks = std::vector<std::pair<size_t, size_t>>;

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

void check_plan(const std::vector<size_t>& offsets, bool core, bool symmetric, size_t budget, size_t nocc = 3) {
    constexpr size_t nbf = 10, nthreads = 2, pairs = 100;
    const size_t ao_size = offsets.back() * pairs;
    Blocks blocks;
    const auto sizes = psi::detail::dfhelper_jk_blocks(
        blocks, offsets, nbf, nthreads, nocc, symmetric, core, ao_size, pairs, budget);
    size_t next = 0, largest_aux = 0, largest_ao = 0;
    for (const auto& block : blocks) {
        require(block.first == next && block.second >= block.first, "Empty or discontinuous block");
        require(block.second + 1 < offsets.size(), "Shell index out of range");
        const size_t aux = offsets[block.second + 1] - offsets[block.first];
        const size_t ao = core ? ao_size : pairs * aux;
        const size_t work = nbf * nocc * aux + nthreads * nbf * std::max(nbf, nocc) +
                            (symmetric ? nbf * nbf : nbf * nocc * aux);
        require(ao + work <= budget, "Block exceeds grant: resident AO tensor was omitted");
        next = block.second + 1;
        largest_aux = std::max(largest_aux, aux);
        largest_ao = std::max(largest_ao, ao);
    }
    require(next + 1 == offsets.size(), "Not all shells were covered");
    require(std::get<0>(sizes) == largest_ao, "Returned AO-buffer extent is wrong");
    require(std::get<1>(sizes) == largest_aux, "Returned auxiliary extent is wrong");
}

void expect_rejection(const std::vector<size_t>& offsets, bool core, size_t budget) {
    Blocks blocks;
    try {
        psi::detail::dfhelper_jk_blocks(blocks, offsets, 10, 2, 3, true, core, offsets.back() * 100, 100, budget);
    } catch (const std::runtime_error& error) {
        require(std::string(error.what()).find("not enough memory") != std::string::npos,
                "Unexpected planner exception");
        return;
    }
    throw std::runtime_error("An indivisible shell exceeding the grant was accepted");
}

int main() {
    try {
        // Several core blocks: the persistent tensor must count in every one.
        const std::vector<size_t> uniform{0, 2, 4, 6, 8, 10, 12, 14, 16};
        check_plan(uniform, true, true, 2020);
        check_plan(uniform, true, false, 2040);
        check_plan(uniform, false, true, 1420);
        check_plan(uniform, false, false, 1440);
        check_plan(uniform, true, true, 2300, 10);
        check_plan(uniform, true, false, 2600, 10);
        check_plan(uniform, true, true, 2900, 20);
        check_plan(uniform, true, false, 3600, 20);
        check_plan(uniform, false, true, 1700, 20);
        check_plan(uniform, false, false, 2400, 20);

        // Exact-fit, single-shell, and large-budget/single-block boundaries.
        for (bool core : {false, true}) {
            check_plan({0, 2}, core, true, 560);
            check_plan(uniform, core, true, 10000);
            expect_rejection({0, 2}, core, 559);
            expect_rejection({0, 2, 20}, core, core ? 2420 : 1000);
        }

        // Vary shell sizes and budgets without relying on a particular partition.
        const std::vector<size_t> irregular{0, 1, 4, 6, 9, 10, 16};
        for (bool core : {false, true}) {
            for (bool symmetric : {false, true}) {
                for (size_t budget = 2200; budget <= 4000; budget += 7) {
                    check_plan(irregular, core, symmetric, budget);
                }
            }
        }
        std::cout << "PASS: DFHelper JK core/disk block budgets, coverage and boundaries\n";
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}

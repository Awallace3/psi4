// Tiny genuine parser/schedule executable; never links a device library.
#include "input.h"
#include <iostream>
struct Fake {
    void upstream_create() { std::cout << "up+\n"; }
    void upstream_destroy() { std::cout << "up-\n"; }
    void plan_create(char p) { std::cout << p << "+\n"; }
    void plan_destroy() { std::cout << "plan-\n"; }
    void compute(char p, int n) { std::cout << p << n << "\n"; }
};
int main(int argc, char** argv) {
    try {
        native_plan::need(argc >= 2, "arguments");
        if (std::string(argv[1]) == "--schedule") {
            Fake f; native_plan::schedule(f); return 0;
        }
        if (std::string(argv[1]) == "--bits") {
            native_plan::need(argc==3,"bits arguments");
            auto in=native_plan::parse(argv[2]);
            auto bits=[](double d) { uint64_t u; std::memcpy(&u,&d,8); std::cout<<u<<"\n"; };
            for (double d:in.xyz) bits(d);
            for (const auto& b:in.bases) for (const auto& s:b.shells) {
                for (double d:s.xyz) bits(d);
                for (double d:s.exp) bits(d);
                for (double d:s.coef) bits(d);
            }
            return 0;
        }
        auto in = native_plan::parse(argv[1]);
        if (argc == 3) native_plan::density(argv[2], in.bases[0].nbf);
        std::cout << in.natom << " " << in.bases[0].nbf << " " << in.bases[1].nbf << "\n";
        return 0;
    } catch (const std::exception& e) { std::cerr << e.what() << "\n"; return 1; }
}

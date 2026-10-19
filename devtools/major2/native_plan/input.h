// Owned transport parser: usable without CUDA or cuEST in host tests.
#pragma once
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace native_plan {
inline void need(bool ok, const char* msg) {
    if (!ok) throw std::runtime_error(msg);
}
struct Shell {
    uint64_t index, center, offset, pure, am;
    std::array<double, 3> xyz;
    std::vector<double> exp, coef;
};
struct Basis {
    uint64_t nbf;
    std::vector<uint64_t> counts;
    std::vector<Shell> shells;
};
struct Input {
    uint64_t natom;
    std::vector<double> xyz;
    std::array<Basis, 2> bases;
};
class Reader {
    std::ifstream f;
    uint64_t remaining;
 public:
    explicit Reader(const std::string& path) : f(path, std::ios::binary | std::ios::ate) {
        need(bool(f), "input open");
        auto n = f.tellg();
        need(n >= 0 && n <= 128*1024*1024, "input byte cap");
        remaining = static_cast<uint64_t>(n);
        f.seekg(0);
        uint16_t one = 1;
        need(*reinterpret_cast<char*>(&one) == 1 && sizeof(double) == 8 &&
             std::numeric_limits<double>::is_iec559, "little endian IEEE754 required");
    }
    void bytes(void* p, uint64_t n) {
        need(n <= remaining, "truncated input");
        need(bool(f.read(static_cast<char*>(p), n)), "input read");
        remaining -= n;
    }
    uint64_t u() { uint64_t x; bytes(&x, 8); return x; }
    double d() {
        double x; bytes(&x, 8);
        need(std::isfinite(x), "nonfinite input");
        return x;
    }
    void end() { need(remaining == 0, "trailing input"); }
};
inline Input parse(const std::string& path) {
    Reader r(path);
    char magic[8]; r.bytes(magic, 8);
    need(std::memcmp(magic, "NDFPLAN1", 8) == 0, "input magic/version");
    Input in{};
    in.natom = r.u();
    need(in.natom > 0 && in.natom <= 512, "natom limit");
    for (auto& b : in.bases) {
        b.nbf = r.u();
        need(b.nbf > 0 && b.nbf <= 16384, "nbf limit");
    }
    for (uint64_t i = 0; i < 3*in.natom; ++i) in.xyz.push_back(r.d());
    uint64_t primitives = 0;
    for (auto& b : in.bases) {
        auto nshell = r.u();
        need(nshell > 0 && nshell <= 32768, "shell limit");
        uint64_t total = 0;
        for (uint64_t a = 0; a < in.natom; ++a) {
            auto n = r.u();
            need(n > 0 && n <= 512, "shells per center limit");
            b.counts.push_back(n); total += n;
        }
        need(total == nshell, "shell total mismatch");
        uint64_t offset = 0;
        for (uint64_t a = 0; a < in.natom; ++a) {
            for (uint64_t q = 0; q < b.counts[a]; ++q) {
                Shell s{};
                s.index=r.u(); s.center=r.u(); s.offset=r.u(); s.pure=r.u(); s.am=r.u();
                auto np = r.u();
                need(s.index == b.shells.size() && s.center == a && s.offset == offset,
                     "shell/AO ordering");
                need(s.pure == 1 && s.am <= 10 && np > 0 && np <= 256, "shell bounds");
                primitives += np;
                need(primitives <= 2000000, "total primitive cap");
                for (auto& x : s.xyz) x=r.d();
                for (uint64_t k=0; k<np; ++k) {
                    auto x=r.d(); need(x > 0, "nonpositive exponent"); s.exp.push_back(x);
                }
                for (uint64_t k=0; k<np; ++k) s.coef.push_back(r.d());
                offset += 2*s.am+1;
                b.shells.push_back(std::move(s));
            }
        }
        need(offset == b.nbf, "nbf/shell mismatch");
    }
    r.end();
    return in;
}
inline std::vector<double> density(const std::string& path, uint64_t nbf) {
    Reader r(path);
    need(nbf > 0 && nbf <= 4096, "density nbf cap");
    std::vector<double> d(nbf*nbf);
    for (auto& x : d) x=r.d();
    r.end();
    return d;
}
// Same schedule compiled into native executable and exercised by fake host backend.
template<class Backend> void schedule(Backend& b) {
    b.upstream_create();
    for (char plan : {'A','B','C'}) {
        if (plan == 'C') { b.upstream_destroy(); b.upstream_create(); }
        b.plan_create(plan);
        b.compute(plan, 1);
        b.compute(plan, 2);
        b.plan_destroy();
    }
    b.upstream_destroy();
}
}

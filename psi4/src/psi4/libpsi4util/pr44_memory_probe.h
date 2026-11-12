// EXPERIMENT ONLY: allocation/phase probes; not part of the production PR.
#pragma once
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <initializer_list>
#include <string>
#include <utility>
#include <sys/resource.h>
#include <unistd.h>

namespace psi {
namespace pr44_probe {
inline void record(const std::string& phase, const char* event,
                   std::initializer_list<std::pair<const char*, size_t>> fields = {}) {
    if (!std::getenv("PR44_MEMORY_TRACE")) return;
    size_t pages = 0, resident = 0;
    FILE* stat = std::fopen("/proc/self/statm", "r");
    if (stat) {
        if (std::fscanf(stat, "%zu %zu", &pages, &resident) != 2) resident = 0;
        std::fclose(stat);
    }
    struct rusage usage {};
    getrusage(RUSAGE_SELF, &usage);
    double now = std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
    std::fprintf(stderr, "PR44_MEM {\"time\":%.6f,\"phase\":\"%s\",\"event\":\"%s\","
                 "\"rss_bytes\":%zu,\"peak_rss_bytes\":%zu",
                 now, phase.c_str(), event, resident * static_cast<size_t>(sysconf(_SC_PAGESIZE)),
                 static_cast<size_t>(usage.ru_maxrss) * 1024);
    for (const auto& field : fields) std::fprintf(stderr, ",\"%s\":%zu", field.first, field.second);
    std::fprintf(stderr, "}\n");
    std::fflush(stderr);
}
struct Scope {
    std::string phase;
    explicit Scope(std::string name) : phase(std::move(name)) { record(phase, "begin"); }
    ~Scope() { record(phase, "end"); }
};
}
}

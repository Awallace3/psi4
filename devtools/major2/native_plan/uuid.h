// CUDA-free byte formatter, shared by production and host-only regression.
#pragma once
#include <iomanip>
#include <sstream>
#include <string>

namespace native_plan {
inline std::string format_uuid(const char* bytes) {
    std::ostringstream uuid;
    uuid << "GPU-" << std::hex << std::setfill('0');
    for (int k=0;k<16;++k) {
        if (k==4 || k==6 || k==8 || k==10) uuid << "-";
        uuid << std::setw(2) << static_cast<unsigned>(static_cast<unsigned char>(bytes[k]));
    }
    return uuid.str();
}
}

#include <hdf5.h>
#include <dlfcn.h>
#include <cstdio>

int main() {
    unsigned major = 0, minor = 0, release = 0;
    if (H5get_libversion(&major,&minor,&release) < 0) return 2;
    Dl_info info{};
    if (!dladdr(reinterpret_cast<void*>(&H5get_libversion),&info) || !info.dli_fname) return 3;
    std::printf("header=%d.%d.%d\nruntime=%u.%u.%u\nlibrary=%s\n",
                H5_VERS_MAJOR,H5_VERS_MINOR,H5_VERS_RELEASE,major,minor,release,info.dli_fname);
    return major == H5_VERS_MAJOR && minor == H5_VERS_MINOR && release == H5_VERS_RELEASE ? 0 : 4;
}

// Standalone diagnostic, public C API only. No Psi4/Python link or import.
// Abort on any failure, with no retry/fallback; process exit reclaims incomplete resources.
#include "input.h"
#include <cuest.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>

using namespace native_plan;
static void cu(cuestStatus_t x, const char* op) {
    if (x != CUEST_STATUS_SUCCESS) throw std::runtime_error(std::string(op)+": cuEST "+std::to_string(x));
}
static void ca(cudaError_t x, const char* op) {
    if (x != cudaSuccess) throw std::runtime_error(std::string(op)+": CUDA "+cudaGetErrorString(x));
}
#define CU(x) cu((x), #x)
#define CA(x) ca((x), #x)
#define BL(x) need((x) == CUBLAS_STATUS_SUCCESS, #x)
#define SO(x) need((x) == CUSOLVER_STATUS_SUCCESS, #x)
static void boundary() { CA(cudaDeviceSynchronize()); }
static void save(const std::string& p, const void* data, size_t n) {
    int fd = open(p.c_str(), O_WRONLY|O_CREAT|O_EXCL, 0444);
    need(fd >= 0, "exclusive output create");
    const char* s=static_cast<const char*>(data);
    while (n) {
        auto k=write(fd, s, n);
        need(k > 0, "output write"); s+=k; n-=k;
    }
    need(fsync(fd) == 0 && close(fd) == 0, "output fsync/close");
    auto slash=p.find_last_of('/');
    int dir=open(p.substr(0,slash).c_str(),O_RDONLY|O_DIRECTORY);
    need(dir>=0 && fsync(dir)==0 && close(dir)==0,"output directory fsync");
}
static void text(const std::string& p, const std::string& s) { save(p, s.data(), s.size()); }
static void env_check() {
    extern char** environ;
    for (char** p=environ; *p; ++p) {
        std::string key(*p); key=key.substr(0,key.find('='));
        bool bad=false;
        for (const auto* pre : {"CUBLAS","CUDNN","CUSOLVER","CUSPARSE","CUEST",
                               "NVIDIA_TF32","TORCH_ALLOW_TF32","CUDA_MATH_","NVIDIA_TF32_OVERRIDE"})
            if (key.rfind(pre,0)==0) bad=true;
        for (const auto* exact : {"CUDA_LAUNCH_BLOCKING","CUDA_DEVICE_MAX_CONNECTIONS",
                                  "CUDA_DEVICE_MAX_COPY_CONNECTIONS","NVIDIA_TF32_OVERRIDE",
                                  "OMP_PROC_BIND","OMP_PLACES","MKL_CBWR","MKL_DEBUG_CPU_TYPE"})
            if (key==exact) bad=true;
        if (key.rfind("CUDA_",0)==0 && key!="CUDA_VISIBLE_DEVICES" && key!="CUDA_HOME" &&
            key!="CUDA_PATH" && key!="CUDA_CACHE_PATH" && key!="CUDA_CACHE_DISABLE" &&
            key!="CUDA_CACHE_MAXSIZE") bad=true;
        need(!bad, "inherited numerical override: stop");
    }
}
struct Workspace {
    cuestWorkspace_t w{};
    void alloc(cuestWorkspaceDescriptor_t d) {
        need(!w.hostBuffer && !w.deviceBuffer, "workspace overlap");
        w.hostBufferSizeInBytes=d.hostBufferSizeInBytes;
        w.deviceBufferSizeInBytes=d.deviceBufferSizeInBytes;
        if (d.hostBufferSizeInBytes) {
            w.hostBuffer=reinterpret_cast<uintptr_t>(std::malloc(d.hostBufferSizeInBytes));
            need(w.hostBuffer, "malloc workspace");
        }
        if (d.deviceBufferSizeInBytes) {
            void* p=nullptr; CA(cudaMalloc(&p,d.deviceBufferSizeInBytes));
            w.deviceBuffer=reinterpret_cast<uintptr_t>(p);
        }
        // SDK example: no workspace initialization. Never zero/poison here.
    }
    void close() {
        if (w.deviceBuffer) CA(cudaFree(reinterpret_cast<void*>(w.deviceBuffer)));
        std::free(reinterpret_cast<void*>(w.hostBuffer)); w={};
    }
};
struct Replay {
    const Input& in;
    std::string out;
    cuestHandle_t h{};
    cuestAOBasis_t bases[2]{};
    Workspace basis_ws[2], pair_ws, plan_ws;
    cuestAOPairList_t pair{};
    cuestDFIntPlan_t plan{};
    cuestDFCoulombComputeParameters_t jp{};
    double *dd=nullptr,*jj=nullptr;
    std::vector<double> j;
    std::ostringstream events;
    std::string uuid;
    Replay(const Input& i, std::string path, cuestHandle_t handle,
           const std::vector<double>& d, std::string id) :
        in(i),out(std::move(path)),h(handle),j(d.size()),uuid(std::move(id)) {
        CA(cudaMalloc(reinterpret_cast<void**>(&dd), d.size()*8));
        CA(cudaMalloc(reinterpret_cast<void**>(&jj), d.size()*8));
        CA(cudaMemcpy(dd,d.data(),d.size()*8,cudaMemcpyHostToDevice)); boundary();
        CU(cuestParametersCreate(CUEST_DFCOULOMBCOMPUTE_PARAMETERS, reinterpret_cast<void**>(&jp)));
        // Native math mode controls FP64; leave JIT/ffloat compute parameters at SDK defaults,
        // exactly as the wrapper. No math-mode intervention cohort.
    }
    void tenants() {
        // UUID is formatted directly from CUDA bytes, not shell input.
        const std::string cmd="nvidia-smi --id="+uuid+
            " --query-compute-apps=gpu_uuid,pid --format=csv,noheader,nounits";
        FILE* pipe=popen(cmd.c_str(),"r"); need(pipe,"nvidia-smi open");
        char line[512]; std::string data;
        while (fgets(line,sizeof(line),pipe)) {
            data+=line; need(data.size()<4096,"tenant output bound");
        }
        need(pclose(pipe)==0,"nvidia-smi status");
        std::istringstream stream(data); unsigned rows=0;
        while (std::getline(stream,data)) {
            if (data.empty()) continue;
            auto comma=data.find(',');
            need(comma!=std::string::npos && data.substr(0,comma)==uuid,"tenant UUID");
            std::istringstream pid(data.substr(comma+1)); long id=0; std::string extra;
            need(bool(pid>>id) && !(pid>>extra) && id==getpid(),"exclusive native PID");
            ++rows;
        }
        need(rows==1,"one native GPU holder");
    }
    void upstream_create() {
        tenants();
        events << "up+\n";
        for (int k=0;k<2;++k) {
            const auto& b=in.bases[k];
            cuestAOShellParameters_t sp{};
            CU(cuestParametersCreate(CUEST_AOSHELL_PARAMETERS,reinterpret_cast<void**>(&sp)));
            std::vector<cuestAOShell_t> shells;
            for (const auto& s:b.shells) {
                cuestAOShell_t sh{};
                CU(cuestAOShellCreate(h,s.pure,s.am,s.exp.size(),s.exp.data(),s.coef.data(),sp,&sh));
                shells.push_back(sh);
            }
            cuestAOBasisParameters_t bp{};
            CU(cuestParametersCreate(CUEST_AOBASIS_PARAMETERS,reinterpret_cast<void**>(&bp)));
            cuestWorkspaceDescriptor_t p{},t{};
            CU(cuestAOBasisCreateWorkspaceQuery(h,in.natom,b.counts.data(),shells.data(),bp,&p,&t,nullptr));
            basis_ws[k].alloc(p); Workspace tmp; tmp.alloc(t);
            CU(cuestAOBasisCreate(h,in.natom,b.counts.data(),shells.data(),bp,&basis_ws[k].w,&tmp.w,&bases[k]));
            boundary(); tmp.close();
            for (auto s:shells) CU(cuestAOShellDestroy(s));
            CU(cuestParametersDestroy(CUEST_AOBASIS_PARAMETERS,bp));
            CU(cuestParametersDestroy(CUEST_AOSHELL_PARAMETERS,sp));
        }
        cuestAOPairListParameters_t pp{};
        CU(cuestParametersCreate(CUEST_AOPAIRLIST_PARAMETERS,reinterpret_cast<void**>(&pp)));
        cuestWorkspaceDescriptor_t p{},t{};
        CU(cuestAOPairListCreateWorkspaceQuery(h,bases[0],in.natom,in.xyz.data(),1.e-12,pp,&p,&t,nullptr));
        pair_ws.alloc(p); Workspace tmp; tmp.alloc(t);
        CU(cuestAOPairListCreate(h,bases[0],in.natom,in.xyz.data(),1.e-12,pp,&pair_ws.w,&tmp.w,&pair));
        boundary(); tmp.close(); CU(cuestParametersDestroy(CUEST_AOPAIRLIST_PARAMETERS,pp));
    }
    void upstream_destroy() {
        need(!plan,"plan must die first"); boundary();
        CU(cuestAOPairListDestroy(pair)); pair=nullptr; boundary(); pair_ws.close();
        for (int k=0;k<2;++k) {
            CU(cuestAOBasisDestroy(bases[k])); bases[k]=nullptr; boundary(); basis_ws[k].close();
        }
        boundary(); events << "up-\n";
    }
    void plan_create(char label) {
        tenants();
        need(!plan,"overlapping plan"); boundary();
        cuestDFIntPlanParameters_t dp{};
        CU(cuestParametersCreate(CUEST_DFINTPLAN_PARAMETERS,reinterpret_cast<void**>(&dp)));
        const double alpha=1., beta=0., omega=0.;
        CU(cuestParametersConfigure(CUEST_DFINTPLAN_PARAMETERS,dp,
           CUEST_DFINTPLAN_PARAMETERS_EXCHANGE_FRACTION,&alpha,sizeof(alpha)));
        CU(cuestParametersConfigure(CUEST_DFINTPLAN_PARAMETERS,dp,
           CUEST_DFINTPLAN_PARAMETERS_LRC_EXCHANGE_FRACTION,&beta,sizeof(beta)));
        CU(cuestParametersConfigure(CUEST_DFINTPLAN_PARAMETERS,dp,
           CUEST_DFINTPLAN_PARAMETERS_LRC_EXCHANGE_OMEGA,&omega,sizeof(omega)));
        cuestWorkspaceDescriptor_t p{},t{};
        CU(cuestDFIntPlanCreateWorkspaceQuery(h,bases[0],bases[1],pair,dp,&p,&t,nullptr));
        events << label << "+\n" << "workspace " << p.hostBufferSizeInBytes << " "
               << p.deviceBufferSizeInBytes << " " << t.hostBufferSizeInBytes << " "
               << t.deviceBufferSizeInBytes << "\n";
        plan_ws.alloc(p); Workspace tmp; tmp.alloc(t);
        CU(cuestDFIntPlanCreate(h,bases[0],bases[1],pair,dp,&plan_ws.w,&tmp.w,&plan));
        boundary(); tmp.close(); CU(cuestParametersDestroy(CUEST_DFINTPLAN_PARAMETERS,dp));
    }
    void plan_destroy() {
        boundary(); CU(cuestDFIntPlanDestroy(plan)); plan=nullptr; boundary();
        plan_ws.close(); events << "plan-\n";
    }
    void compute(char label, int call) {
        env_check(); tenants(); boundary();
        cuestMathMode_t mode;
        CU(cuestGetMathMode(h,&mode)); need(mode==CUEST_NATIVE_FP64_MATH_MODE,"FP64 mode");
        cuestWorkspaceDescriptor_t t{};
        CU(cuestDFCoulombComputeWorkspaceQuery(h,plan,jp,&t,dd,jj));
        Workspace tmp; tmp.alloc(t);
        CU(cuestDFCoulombCompute(h,plan,jp,&tmp.w,dd,jj)); boundary();
        CA(cudaMemcpy(j.data(),jj,j.size()*8,cudaMemcpyDeviceToHost)); boundary();
        for (auto x:j) need(std::isfinite(x),"nonfinite J");
        save(out+"/"+label+std::to_string(call)+".bin",j.data(),j.size()*8);
        tmp.close(); events << label << call << "\n";
    }
    void close(const std::vector<double>& d) {
        CA(cudaMemcpy(j.data(),dd,j.size()*8,cudaMemcpyDeviceToHost)); boundary();
        need(std::memcmp(j.data(),d.data(),j.size()*8)==0,"density changed by SDK");
        CU(cuestParametersDestroy(CUEST_DFCOULOMBCOMPUTE_PARAMETERS,jp));
        CA(cudaFree(dd)); CA(cudaFree(jj)); boundary();
        text(out+"/schedule.txt",events.str());
    }
};
int main(int argc, char** argv) {
    try {
        need(argc==5,"usage: native basis.bin D.bin output-dir expected-GPU-UUID");
        env_check();
        const char* job=std::getenv("SLURM_JOB_ID"); need(job && *job,"allocation required");
        text(std::string(argv[3])+"/effective-env.json",
             "{\"numerical_overrides\":{},\"pid\":"+std::to_string(getpid())+"}\n");
        auto in=parse(argv[1]);
        need(in.natom==157 && in.bases[0].nbf==1786 && in.bases[1].nbf==7265,"frozen dimensions");
        auto d=density(argv[2],1786);
        int count=0,device=-1,rv=0;
        CA(cudaGetDeviceCount(&count)); CA(cudaGetDevice(&device));
        need(count==1 && device==0,"one logical GPU required");
        cudaDeviceProp props{}; CA(cudaGetDeviceProperties(&props,0));
        cudaUUID_t u{}; CA(cudaDeviceGetUuid(&u,0)); CA(cudaRuntimeGetVersion(&rv));
        std::ostringstream uuid; uuid << "GPU-" << std::hex << std::setfill('0');
        for (int k=0;k<16;++k) {
            if (k==4 || k==6 || k==8 || k==10) uuid << "-";
            uuid << std::setw(2) << static_cast<unsigned>(static_cast<unsigned char>(u.bytes[k]));
        }
        need(uuid.str()==argv[4] && std::string(props.name).find("H200")!=std::string::npos,"GPU identity");
        need(CUDART_VERSION==13030 && rv==13030 && CUEST_VERSION==250,"header/runtime ABI");
        cudaStream_t stream{}; cublasHandle_t blas{}; cusolverDnHandle_t solver{};
        CA(cudaStreamCreate(&stream)); BL(cublasCreate(&blas)); SO(cusolverDnCreate(&solver));
        BL(cublasSetStream(blas,stream)); SO(cusolverDnSetStream(solver,stream));
        cuestHandleParameters_t hp{};
        CU(cuestParametersCreate(CUEST_HANDLE_PARAMETERS,&hp));
        CU(cuestParametersConfigure(CUEST_HANDLE_PARAMETERS,hp,CUEST_HANDLE_PARAMETERS_CUDASTREAM,&stream,sizeof(stream)));
        CU(cuestParametersConfigure(CUEST_HANDLE_PARAMETERS,hp,CUEST_HANDLE_PARAMETERS_CUBLAS,&blas,sizeof(blas)));
        CU(cuestParametersConfigure(CUEST_HANDLE_PARAMETERS,hp,CUEST_HANDLE_PARAMETERS_CUSOLVER,&solver,sizeof(solver)));
        cuestHandle_t h{}; CU(cuestCreate(hp,&h)); boundary();
        CU(cuestParametersDestroy(CUEST_HANDLE_PARAMETERS,hp));
        uint32_t major=0,minor=0,patch=0;
        CU(cuestGetMajorVersion(h,&major)); CU(cuestGetMinorVersion(h,&minor)); CU(cuestGetPatchVersion(h,&patch));
        need(major==0 && minor==2 && patch==50,"SDK runtime version");
        CU(cuestSetMathMode(h,CUEST_NATIVE_FP64_MATH_MODE)); boundary();
        Replay b(in,argv[3],h,d,uuid.str()); schedule(b); b.close(d);
        // Verify input device density remained unchanged (not a C-reconstruction).
        // All scientific output is raw FP64, not any opaque workspace.
        std::ifstream maps("/proc/self/maps");
        need(bool(maps),"runtime mapping read");
        std::string maptext((std::istreambuf_iterator<char>(maps)),{});
        text(std::string(argv[3])+"/maps.txt",maptext);
        CU(cuestDestroy(h)); boundary(); SO(cusolverDnDestroy(solver)); BL(cublasDestroy(blas));
        CA(cudaStreamDestroy(stream)); boundary();
        text(std::string(argv[3])+"/native-complete.json",
             "{\"outputs\":6,\"plans\":3,\"handles\":1,\"upstream_builds\":2,\"sdk\":\"0.2.50\","
             "\"cudart\":13030,\"pid\":"+std::to_string(getpid())+",\"uuid\":\""+uuid.str()+"\",\"job\":\""+job+"\"}\n");
        return 0;
    } catch(const std::exception& e) { std::cerr << e.what() << "\n"; return 1; }
}

"""FP64 NVRTC source for the explicitly selected Python/cuEST CUDA backend.

No CUDA work occurs at import. No fast-math, denominator clamp or atomics.
"""

CUDA_SOURCE = r'''
extern "C" {
__device__ __forceinline__
double factor(const double* p, long long q, long long v, long long o,
              long long Q, long long V, long long O, int fortran_order) {
    return p[fortran_order ? q + Q*(v + V*o) : (q*V + v)*O + o];
}

// Each factor field has logical shape[Q,V,O]; packed output is C-order[K,V*O].
// Optional extra fields fuse Dar+Ear / Dbs+Ebs without a full-size temporary.
__global__
void pack_factors(long long Q, long long V, long long O, int fortran_order,
                  const double* first, const double* first_extra,
                  const double* second, const double* second_extra,
                  const double* tail1, const double* tail2, double* packed) {
    const long long columns = V*O;
    const long long total = (2*Q+2)*columns;
    for (long long index = (long long)blockIdx.x*blockDim.x + threadIdx.x;
         index < total; index += (long long)blockDim.x*gridDim.x) {
        const long long row = index/columns, col = index%columns;
        const long long v = col/O, o = col%O;
        double value;
        if (row < Q) {
            value = factor(first,row,v,o,Q,V,O,fortran_order);
            if (first_extra) value += factor(first_extra,row,v,o,Q,V,O,fortran_order);
        } else if (row < 2*Q) {
            value = factor(second,row-Q,v,o,Q,V,O,fortran_order);
            if (second_extra) value += factor(second_extra,row-Q,v,o,Q,V,O,fortran_order);
        } else {
            value = (row == 2*Q ? tail1 : tail2)[o*V+v];
        }
        packed[index] = value;
    }
}

// One block owns an occupied pair. r/s block launches are ordered on one stream.
// V,X are C-order[(r,a),(s,b)], W is C-order[(s,a),(r,b)].
__global__
void reduce_dispersion(long long A, long long B, long long R, long long S,
                       long long r0, long long s0,
                       const double* ea, const double* eb,
                       const double* er, const double* es,
                       const double* V, const double* X, const double* W,
                       double* disp, double* exch) {
    __shared__ double ds[128];
    __shared__ double xs[128];
    for (long long pair = blockIdx.x; pair < A*B; pair += gridDim.x) {
        const long long a = pair/B, b = pair%B;
        double d = 0.0, x = 0.0;
        for (long long rs = threadIdx.x; rs < R*S; rs += blockDim.x) {
            const long long r = rs/S, s = rs%S;
            const long long i = (r*A+a)*(S*B) + s*B+b;
            const long long j = (s*A+a)*(R*B) + r*B+b;
            const double t = V[i]/(ea[a]+eb[b]-er[r0+r]-es[s0+s]);
            d += 4.0*t*V[i];
            x -= 2.0*t*(X[i]+W[j]);
        }
        ds[threadIdx.x] = d;
        xs[threadIdx.x] = x;
        __syncthreads();
        for (unsigned int stride = blockDim.x/2; stride; stride /= 2) {
            if (threadIdx.x < stride) {
                ds[threadIdx.x] += ds[threadIdx.x+stride];
                xs[threadIdx.x] += xs[threadIdx.x+stride];
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            disp[pair] += ds[0];
            exch[pair] += xs[0];
        }
        __syncthreads();
    }
}
}
'''

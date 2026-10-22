"""FP64 NVRTC source for the explicitly selected Python/cuEST CUDA backend.

No CUDA work occurs at import. No fast-math, denominator clamp or atomics.
"""

CUDA_SOURCE = r'''
extern "C" {
__global__
void inverse_sqrt_weights(long long N, const double* eigenvalues, const double* policy,
                          double* weights, double* stats) {
    __shared__ double maximum[128];
    __shared__ double bad[128];
    double largest=0.0, invalid=0.0;
    for (long long i=threadIdx.x; i<N; i+=blockDim.x) {
        double value=eigenvalues[i];if (value<0.0) value=-value;
        if (!(value<1.e300)) invalid=1.0;
        else if (value>largest) largest=value;
    }
    maximum[threadIdx.x]=largest;bad[threadIdx.x]=invalid;
    __syncthreads();
    for (unsigned int stride=blockDim.x/2; stride; stride/=2) {
        if (threadIdx.x<stride) {
            if (maximum[threadIdx.x+stride]>maximum[threadIdx.x])
                maximum[threadIdx.x]=maximum[threadIdx.x+stride];
            if (bad[threadIdx.x+stride]>bad[threadIdx.x]) bad[threadIdx.x]=bad[threadIdx.x+stride];
        }
        __syncthreads();
    }
    // CPU Matrix::power uses STRICT |lambda| < condition*max|lambda|,
    // then replaces nonfinite negative/zero inverse powers by zero.
    double rank=0.0;
    for (long long i=threadIdx.x; i<N; i+=blockDim.x) {
        const double value=eigenvalues[i];
        double weight=0.0;
        if (value>0.0 && !(value<policy[0]*maximum[0]) && value<1.e300)
            weight=1.0/sqrt(value);
        weights[i]=weight;
        if (weight>0.0) rank+=1.0;
    }
    // maximum[0] is still read by all threads in the loop above.
    __syncthreads();
    maximum[threadIdx.x]=rank;
    __syncthreads();
    for (unsigned int stride=blockDim.x/2; stride; stride/=2) {
        if (threadIdx.x<stride) maximum[threadIdx.x]+=maximum[threadIdx.x+stride];
        __syncthreads();
    }
    if (threadIdx.x==0) {
        stats[0]=maximum[0];stats[1]=bad[0];
    }
}

// cuSOLVER writes column-major eigenvectors. Viewed as C-order this is V.T;
// scale its rows, then V diag(weights) V.T = gemm_at_b(V.T,scaled_V.T).
__global__
void scale_eigenvector_rows(long long N, const double* transpose_vectors,
                            const double* weights, double* output) {
    for (long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;
         i<N*N; i+=(long long)blockDim.x*gridDim.x)
        output[i]=weights[i/N]*transpose_vectors[i];
}

// Positive singlet RHF H+ = gap + 4J - K - K^T in canonical MO coordinates.
// This bounded reference kernel fuses both distinct exchange contractions.
__global__
void rhf_response_product(long long Q, long long O, long long V,
                          const double* ov, const double* oo, const double* vv,
                          const double* trial, const double* eo, const double* ev,
                          const double* coulomb, double* output) {
    __shared__ double sums[128];
    for (long long ia=blockIdx.x; ia<O*V; ia+=gridDim.x) {
        const long long i=ia/V, a=ia%V;
        double value=0.0;
        for (long long qjb=threadIdx.x; qjb<Q*O*V; qjb+=blockDim.x) {
            const long long b=qjb%V, j=(qjb/V)%O, q=qjb/(V*O);
            value+=(oo[(q*O+i)*O+j]*vv[(q*V+a)*V+b]
                    +ov[(q*O+i)*V+b]*ov[(q*O+j)*V+a])*trial[j*V+b];
        }
        sums[threadIdx.x]=value;
        __syncthreads();
        for (unsigned int stride=blockDim.x/2; stride; stride/=2) {
            if (threadIdx.x<stride) sums[threadIdx.x]+=sums[threadIdx.x+stride];
            __syncthreads();
        }
        if (threadIdx.x==0)
            output[ia]=(ev[a]-eo[i])*trial[ia]+4.0*coulomb[ia]-sums[0];
        __syncthreads();
    }
}

__global__
void response_dot(long long count, const double* left, const double* right, double* output) {
    __shared__ double sums[128];
    double value=0.0;
    for (long long i=threadIdx.x; i<count; i+=blockDim.x) value+=left[i]*right[i];
    sums[threadIdx.x]=value;
    __syncthreads();
    for (unsigned int stride=blockDim.x/2; stride; stride/=2) {
        if (threadIdx.x<stride) sums[threadIdx.x]+=sums[threadIdx.x+stride];
        __syncthreads();
    }
    if (threadIdx.x==0) output[0]=sums[0];
}

__global__
void response_precondition(long long O, long long V, const double* eo,
                           const double* ev, const double* residual, double* output) {
    for (long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;
         i<O*V; i+=(long long)blockDim.x*gridDim.x)
        output[i]=residual[i]/(ev[i%V]-eo[i/V]);
}

__global__
void response_step(long long count, const double* rho, const double* curvature,
                    const double* direction, const double* product,
                    double* solution, double* residual) {
    const double alpha=rho[0]/curvature[0];
    for (long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;
         i<count; i+=(long long)blockDim.x*gridDim.x) {
        solution[i]+=alpha*direction[i];
        residual[i]-=alpha*product[i];
    }
}

__global__
void response_direction(long long count, const double* rho_new, const double* rho_old,
                         const double* z, double* direction) {
    const double beta=rho_new[0]/rho_old[0];
    for (long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;
         i<count; i+=(long long)blockDim.x*gridDim.x)
        direction[i]=z[i]+beta*direction[i];
}

// Convergence numerics stay on GPU. Host reads only these three scalar controls.
__global__
void response_metrics(const double* residual2, const double* rhs2,
                       const double* tolerance, double* metrics) {
    if (blockIdx.x==0 && threadIdx.x==0) {
        const double r=sqrt(residual2[0]), b=sqrt(rhs2[0]);
        double threshold=tolerance[1]*b;
        if (threshold<tolerance[0]) threshold=tolerance[0];
        metrics[0]=r;
        metrics[1]=b>0.0 ? r/b : r;
        metrics[2]=r<=threshold ? 1.0 : 0.0;
    }
}

// The generalized density is Cprime*Cocc.T; no spin factor or re-whitening.
__global__
void trace_factors(long long Q, long long O, const double* factors, double* output) {
    __shared__ double sums[128];
    for (long long q=blockIdx.x; q<Q; q+=gridDim.x) {
        double value=0.0;
        for (long long o=threadIdx.x; o<O; o+=blockDim.x)
            value+=factors[(q*O+o)*O+o];
        sums[threadIdx.x]=value;
        __syncthreads();
        for (unsigned int stride=blockDim.x/2; stride; stride/=2) {
            if (threadIdx.x<stride) sums[threadIdx.x]+=sums[threadIdx.x+stride];
            __syncthreads();
        }
        if (threadIdx.x==0) output[q]=sums[0];
        __syncthreads();
    }
}

__global__
void matrix_error(long long count, const double* input, double* output) {
    __shared__ double worst[128];
    double value=0.0;
    for (long long i=threadIdx.x; i<count; i+=blockDim.x) {
        double v=input[i];
        if (v<0.0) v=-v;
        if (!(v<1.e300)) v=1.e300;
        if (v>value) value=v;
    }
    worst[threadIdx.x]=value;
    __syncthreads();
    for (unsigned int stride=blockDim.x/2; stride; stride/=2) {
        if (threadIdx.x<stride && worst[threadIdx.x+stride]>worst[threadIdx.x])
            worst[threadIdx.x]=worst[threadIdx.x+stride];
        __syncthreads();
    }
    if (threadIdx.x==0) output[0]=worst[0];
}

__global__
void single_gap_error(long long O, long long V, const double* occupied,
                      const double* virtuals, double* output) {
    __shared__ double bad[128];
    double value=0.0;
    for (long long i=threadIdx.x; i<O*V; i+=blockDim.x) {
        const double o=occupied[i/V], v=virtuals[i%V];
        if (!(o>-1.e300 && o<1.e300 && v>-1.e300 && v<1.e300 && o<v)) value=1.0;
    }
    bad[threadIdx.x]=value;
    __syncthreads();
    for (unsigned int stride=blockDim.x/2; stride; stride/=2) {
        if (threadIdx.x<stride && bad[threadIdx.x+stride]>bad[threadIdx.x])
            bad[threadIdx.x]=bad[threadIdx.x+stride];
        __syncthreads();
    }
    if (threadIdx.x==0) output[0]=bad[0];
}

__global__
void make_single_amplitudes(long long O, long long V, const double* source,
                            const double* occupied, const double* virtuals, double* output) {
    for (long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;
         i<O*V; i+=(long long)blockDim.x*gridDim.x)
        output[i]=source[i]/(occupied[i/V]-virtuals[i%V]);
}

// One block owns an occupied target; one ordered call owns a source column.
__global__
void reduce_induction_source(long long O, long long V, long long C, long long column,
                             const double* amplitudes, const double* w, const double* u,
                             double* induction, double* exchange) {
    __shared__ double ind[128];
    __shared__ double exch[128];
    for (long long o=blockIdx.x; o<O; o+=gridDim.x) {
        double d=0.0, x=0.0;
        for (long long v=threadIdx.x; v<V; v+=blockDim.x) {
            const long long i=o*V+v;
            d+=amplitudes[i]*w[i];
            if (u) x+=amplitudes[i]*u[i];
        }
        ind[threadIdx.x]=d;exch[threadIdx.x]=x;
        __syncthreads();
        for (unsigned int stride=blockDim.x/2; stride; stride/=2) {
            if (threadIdx.x<stride) {
                ind[threadIdx.x]+=ind[threadIdx.x+stride];
                exch[threadIdx.x]+=exch[threadIdx.x+stride];
            }
            __syncthreads();
        }
        if (threadIdx.x==0) {
            induction[o*C+column]=2.0*ind[0];
            if (exchange) exchange[o*C+column]=2.0*exch[0];
        }
        __syncthreads();
    }
}

// Native JKFIT factors Bab[Q,a,b], Bba[Q,b,a]; all occupied, frozen included.
__global__
void reduce_fragment_exchange(long long Q, long long A, long long B,
                              const double* Sab, const double* Sba,
                              const double* WBab, const double* WAba,
                              const double* Bab, const double* Bba, double* output) {
    __shared__ double products[128];
    for (long long pair=blockIdx.x; pair<A*B; pair+=gridDim.x) {
        const long long a=pair/B, b=pair%B;
        double value=0.0;
        for (long long q=threadIdx.x; q<Q; q+=blockDim.x)
            value+=Bab[(q*A+a)*B+b]*Bba[(q*B+b)*A+a];
        products[threadIdx.x]=value;
        __syncthreads();
        for (unsigned int stride=blockDim.x/2; stride; stride/=2) {
            if (threadIdx.x<stride) products[threadIdx.x]+=products[threadIdx.x+stride];
            __syncthreads();
        }
        if (threadIdx.x==0)
            output[pair]=-2.0*(Sab[pair]*WBab[pair]+Sba[b*A+a]*WAba[b*A+a]+products[0]);
        __syncthreads();
    }
}

// C-order [Q,O,O] -> [Q,O]. Retain the native DFMO metric gauge.
__global__
void diagonal_factors(long long Q, long long O, const double* input, double* output) {
    for (long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;
         i<Q*O; i+=(long long)blockDim.x*gridDim.x) {
        const long long q=i/O, o=i%O;
        output[i]=input[(q*O+o)*O+o];
    }
}

// Each output slot has exactly one writer. Different calls use one stream.
__global__
void scatter_matrix(long long rows, long long cols, long long stride,
                    long long row0, long long col0, long long weight,
                    const double* input, double* output) {
    for (long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;
         i<rows*cols; i+=(long long)blockDim.x*gridDim.x)
        output[(row0+i/cols)*stride+col0+i%cols]+=(double)weight*input[i];
}

// Both sides use the full dimer atom index, including zero-charge ghost slots.
// Distinct coincident charged centers deliberately produce Inf, never a clamp.
__global__
void fragment_nuclear(long long N, const double* xyz,
                      const double* za, const double* zb, double* output) {
    for (long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;
         i<N*N; i+=(long long)blockDim.x*gridDim.x) {
        const long long a=i/N, b=i%N;
        double value=0.0;
        if (a!=b && za[a]!=0.0 && zb[b]!=0.0) {
            const double x=xyz[3*a]-xyz[3*b];
            const double y=xyz[3*a+1]-xyz[3*b+1];
            const double z=xyz[3*a+2]-xyz[3*b+2];
            value=za[a]*zb[b]/sqrt(x*x+y*y+z*z);
        }
        output[i]=value;
    }
}

__global__
void reduce_values(long long count, const double* input, double* output) {
    __shared__ double sums[128];
    double value=0.0;
    for (long long i=threadIdx.x; i<count; i+=blockDim.x) value+=input[i];
    sums[threadIdx.x]=value;
    __syncthreads();
    for (unsigned int stride=blockDim.x/2; stride; stride/=2) {
        if (threadIdx.x<stride) sums[threadIdx.x]+=sums[threadIdx.x+stride];
        __syncthreads();
    }
    if (threadIdx.x==0) output[0]=sums[0];
}

// C-order [r,a,s,b] <-> [r,s,b,a], for occupied-only LO rotations.
__global__
void permute_occupied_a(long long R, long long A, long long S, long long B,
                       int reverse, const double* input, double* output) {
    for (long long i = (long long)blockIdx.x*blockDim.x+threadIdx.x;
         i < R*A*S*B; i += (long long)blockDim.x*gridDim.x) {
        const long long b=i%B, s=(i/B)%S, a=(i/(B*S))%A, r=i/(B*S*A);
        const long long j=((r*S+s)*B+b)*A+a;
        if (reverse) output[i]=input[j];
        else output[j]=input[i];
    }
}

// Denominators are diagonal in canonical orbitals, NOT localized orbitals.
__global__
void make_amplitudes(long long A, long long B, long long R, long long S,
                     long long r0, long long s0,
                     const double* ea, const double* eb, const double* er,
                     const double* es, const double* V, double* T) {
    for (long long i = (long long)blockIdx.x*blockDim.x+threadIdx.x;
         i < R*A*S*B; i += (long long)blockDim.x*gridDim.x) {
        const long long b=i%B, s=(i/B)%S, a=(i/(B*S))%A, r=i/(B*S*A);
        T[i]=V[i]/(ea[a]+eb[b]-er[r0+r]-es[s0+s]);
    }
}

__global__
void rotation_error(long long N, const double* gram, double* error) {
    __shared__ double worst[128];
    double value=0.0;
    for (long long i=threadIdx.x; i<N*N; i+=blockDim.x) {
        double v=gram[i]-(i/N==i%N ? 1.0 : 0.0);
        if (v<0.0) v=-v;
        if (!(v<1.e300)) v=1.e300; // NaN/Inf also fail validation.
        if (v>value) value=v;
    }
    worst[threadIdx.x]=value;
    __syncthreads();
    for (unsigned int stride=blockDim.x/2; stride; stride/=2) {
        if (threadIdx.x<stride && worst[threadIdx.x+stride]>worst[threadIdx.x])
            worst[threadIdx.x]=worst[threadIdx.x+stride];
        __syncthreads();
    }
    if (threadIdx.x==0) error[0]=worst[0];
}

__global__
void reduce_localized_dispersion(long long A, long long B, long long R, long long S,
                                const double* T, const double* V, const double* X,
                                const double* W, double* disp, double* exch) {
    __shared__ double ds[128];
    __shared__ double xs[128];
    for (long long pair=blockIdx.x; pair<A*B; pair+=gridDim.x) {
        const long long a=pair/B, b=pair%B;
        double d=0.0, x=0.0;
        for (long long rs=threadIdx.x; rs<R*S; rs+=blockDim.x) {
            const long long r=rs/S, s=rs%S;
            const long long i=(r*A+a)*(S*B)+s*B+b;
            const long long j=(s*A+a)*(R*B)+r*B+b;
            d+=4.0*T[i]*V[i];
            x-=2.0*T[i]*(X[i]+W[j]);
        }
        ds[threadIdx.x]=d; xs[threadIdx.x]=x;
        __syncthreads();
        for (unsigned int stride=blockDim.x/2; stride; stride/=2) {
            if (threadIdx.x<stride) {
                ds[threadIdx.x]+=ds[threadIdx.x+stride];
                xs[threadIdx.x]+=xs[threadIdx.x+stride];
            }
            __syncthreads();
        }
        if (threadIdx.x==0) { disp[pair]+=ds[0]; exch[pair]+=xs[0]; }
        __syncthreads();
    }
}

// Copy a column window, optionally transposed, without staging on the host.
__global__
void matrix_window(long long rows, long long stride, long long start,
                   long long columns, int transpose,
                   const double* input, double* output) {
    for (long long i = (long long)blockIdx.x*blockDim.x + threadIdx.x;
         i < rows*columns; i += (long long)blockDim.x*gridDim.x) {
        const long long row = i/columns, col = i%columns;
        output[transpose ? col*rows+row : i] = input[row*stride+start+col];
    }
}

__global__
void linear_combination(long long count, long long alpha, long long beta,
                        const double* left, const double* right, double* output) {
    for (long long i = (long long)blockIdx.x*blockDim.x + threadIdx.x;
         i < count; i += (long long)blockDim.x*gridDim.x) {
        output[i] = (double)alpha*left[i] + (double)beta*right[i];
    }
}

// Final global signed components, not a host NumPy sum. Launch one 128-thread
// block; downloads are reporting only. No atomics or cross-block race.
__global__
void reduce_components(long long count, const double* disp, const double* exch,
                       double* components) {
    __shared__ double ds[128];
    __shared__ double xs[128];
    double d = 0.0, x = 0.0;
    for (long long i = threadIdx.x; i < count; i += blockDim.x) {
        d += disp[i];
        x += exch[i];
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
        components[0] = ds[0];
        components[1] = xs[0];
        components[2] = ds[0]+xs[0];
    }
}

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

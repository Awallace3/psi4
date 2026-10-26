/* SPDX-License-Identifier: LGPL-3.0-only
 *
 * Uses the GPU-history / small-host-system pattern illustrated by NVIDIA's
 * Apache-2.0 CUDALibrarySamples cuEST DIIS example. Psi4 retains its own
 * conditioned coefficient solver, spin conventions and A/EDIIS algorithms.
 */
#include "cuest_diis.h"
#ifdef USING_cuEST
#include "cuESTCommon.h"
#include "cuest_xc_response.h"
#include "psi4/libmints/matrix.h"
#include "psi4/libpsi4util/PsiOutStream.h"
#include <cublas_v2.h>
#include <algorithm>
#include <cmath>
#include <limits>

extern cudaStream_t stream_handle;
namespace psi {
namespace {
void cuda_check(cudaError_t s) {
    if (s != cudaSuccess) throw PSIEXCEPTION(std::string("cuEST DIIS CUDA: ") + cudaGetErrorString(s));
}
void blas_check(cublasStatus_t s) {
    if (s != CUBLAS_STATUS_SUCCESS)
        throw PSIEXCEPTION("cuEST DIIS cuBLAS status " + std::to_string(static_cast<int>(s)));
}
std::size_t multiply(std::size_t a, std::size_t b) {
    if (a && b > std::numeric_limits<std::size_t>::max() / a)
        throw PSIEXCEPTION("cuEST DIIS: matrix/history size overflow.");
    return a * b;
}
void check_matrix(const SharedMatrix& m, int rows, int cols) {
    if (!m || m->nirrep() != 1 || m->symmetry() || m->rowdim() != rows || m->coldim() != cols)
        throw PSIEXCEPTION("cuEST DIIS requires compatible dense C1 matrices.");
}
struct PendingWork {
    bool done = false;
    ~PendingWork() { if (!done) cudaStreamSynchronize(stream_handle); }
};
}  // namespace

struct CuestDIIS::Impl {
    int requested, memory_mb, cap = 0, count = 0, pivot = -1;
    int nbf = 0, nmo = 0, nspin = 0, ao_stride = 0, error_stride = 0;
    std::size_t iteration = 0;
    bool keep_error, keep_density, oldest, fixed_ready = false;
    cublasHandle_t blas = nullptr;
    CuestDeviceArray s, x, f, d, error, work1, work2, history_f, history_d, history_e, small, coeff;
    std::vector<double> e, q;
    std::map<std::string, double> counters;

    Impl(int vectors, int mb, bool errors, bool densities, bool old)
        : requested(vectors), memory_mb(mb), keep_error(errors), keep_density(densities), oldest(old) {
        if (requested <= 0 || requested > 128 || memory_mb <= 0 || (!errors && !densities))
            throw PSIEXCEPTION("cuEST DIIS requires 1 to 128 vectors, positive memory, and a DIIS or A/EDIIS engine.");
        cuest_common::ensure_cuest_initialized();
        blas_check(cublasCreate(&blas));
        try { blas_check(cublasSetStream(blas, stream_handle)); }
        catch (...) { cublasDestroy(blas); blas = nullptr; throw; }
    }
    ~Impl() {
        // Upload/download sources and reductions may still reference these buffers.
        cudaStreamSynchronize(stream_handle);
        if (blas) cublasDestroy(blas);
    }
    void release() {
        cuda_check(cudaStreamSynchronize(stream_handle));
        for (auto* a : {&s, &x, &f, &d, &error, &work1, &work2, &history_f,
                        &history_d, &history_e, &small, &coeff}) a->release();
        nbf = nmo = nspin = ao_stride = error_stride = cap = count = 0;
        pivot = -1; iteration = 0; fixed_ready = false;
        e.clear(); q.clear();
        counters["allocated_device_bytes"] = 0;
        counters["history_device_bytes"] = 0;
        counters["fixed_workspace_device_bytes"] = 0;
    }
    double* ptr(CuestDeviceArray& a) { return a.ensure(0, "DIIS"); }
    void allocate(CuestDeviceArray& a, std::size_t elements) {
        a.ensure(elements, "DIIS history/workspace");
    }
    void initialize(int n, int r, int spins) {
        if (nbf) {
            if (n != nbf || r != nmo || spins != nspin)
                throw PSIEXCEPTION("cuEST DIIS shape changed; reset the subspace before changing basis/spin dimensions.");
            return;
        }
        if (n <= 0 || r <= 0 || r > n || spins < 1 || spins > 2)
            throw PSIEXCEPTION("cuEST DIIS: invalid SCF dimensions.");
        const auto n2 = multiply(n, n), r2 = multiply(r, r), nx = multiply(n, r);
        const auto ao = multiply(n2, spins), er = multiply(r2, spins);
        if (ao > std::numeric_limits<int>::max() || er > std::numeric_limits<int>::max())
            throw PSIEXCEPTION("cuEST DIIS exceeds cuBLAS vector-length limits.");
        // Fixed S/X, F/D/current error, and two reusable AO transform panels.
        const auto fixed = 3*n2 + nx + 2*ao + er;
        const auto per_entry = ao + (keep_error ? er : 0) + (keep_density ? ao : 0);
        const auto fixed_bytes = multiply(fixed, sizeof(double));
        const auto entry_bytes = multiply(per_entry, sizeof(double));
        const auto metadata_bytes = multiply(multiply(requested, 4), sizeof(double));
        const auto budget = multiply(static_cast<std::size_t>(memory_mb), 1000000);
        std::size_t free = 0, total = 0;
        cuda_check(cudaMemGetInfo(&free, &total));
        // Leave headroom for the rest of SCF/desktop; this is a DIIS-only budget,
        // not a reservation or guarantee for subsequent JK/diagonalization.
        const auto available = std::min(budget, free - free/10);
        if (available <= fixed_bytes || available - fixed_bytes <= metadata_bytes)
            throw PSIEXCEPTION("cuEST DIIS memory limit/free device memory cannot hold its fixed workspace. "
                               "Increase CUEST_DIIS_MAX_MEMORY_MB or retain host DIIS.");
        const auto slots = (available - fixed_bytes - metadata_bytes) / entry_bytes;
        cap = static_cast<int>(std::min<std::size_t>(requested, slots));
        if (cap < std::min(2, requested))
            throw PSIEXCEPTION("cuEST DIIS memory limit/free device memory cannot hold two history entries. "
                               "Increase CUEST_DIIS_MAX_MEMORY_MB or retain host DIIS.");
        allocate(s, n2); allocate(x, nx); allocate(f, ao); allocate(d, ao); allocate(error, er);
        allocate(work1, n2); allocate(work2, n2);
        allocate(history_f, multiply(ao, cap));
        if (keep_error) allocate(history_e, multiply(er, cap));
        if (keep_density) allocate(history_d, multiply(ao, cap));
        allocate(small, 3*static_cast<std::size_t>(requested));
        allocate(coeff, requested);
        nbf = n; nmo = r; nspin = spins; ao_stride = ao; error_stride = er;
        e.assign(multiply(cap, cap), 0.); q.assign(keep_density ? multiply(cap, cap) : 0, 0.);
        const auto bytes = fixed_bytes + entry_bytes*cap + metadata_bytes;
        counters["allocated_device_bytes"] = bytes;
        counters["peak_allocated_device_bytes"] = std::max(counters["peak_allocated_device_bytes"], double(bytes));
        counters["history_device_bytes"] = entry_bytes*cap;
        counters["fixed_workspace_device_bytes"] = fixed_bytes + metadata_bytes;
        counters["effective_max_vectors"] = cap;
        counters["requested_max_vectors"] = requested;
        counters["memory_limit_bytes"] = budget;
        outfile->Printf("    cuEST DIIS: %d/%d history vectors, %.3f GiB device history/workspace "
                        "(limit %.3f GiB).\n", cap, requested,
                        bytes*cuest_common::to_gib, budget*cuest_common::to_gib);
    }
    void upload(const SharedMatrix& matrix, double* out) {
        const auto bytes = multiply(matrix->size(), sizeof(double));
        cuda_check(cudaMemcpyAsync(out, matrix->pointer()[0], bytes, cudaMemcpyHostToDevice, stream_handle));
        counters["matrix_host_to_device_bytes"] += bytes;
    }
    void upload_spins(const std::vector<SharedMatrix>& matrices, CuestDeviceArray& out, int dim) {
        if (matrices.size() != static_cast<std::size_t>(nspin))
            throw PSIEXCEPTION("cuEST DIIS: inconsistent number of spin matrices.");
        for (const auto& matrix : matrices) check_matrix(matrix, dim, dim);
        for (int spin = 0; spin < nspin; ++spin) {
            upload(matrices[spin], ptr(out) + std::size_t(spin)*dim*dim);
        }
    }
    void gemm(bool ta, bool tb, int m, int n, int k, const double* a, const double* b, double* c) {
        const double one = 1., zero = 0.;
        blas_check(cublasDgemm(blas, tb ? CUBLAS_OP_T : CUBLAS_OP_N, ta ? CUBLAS_OP_T : CUBLAS_OP_N,
            n, m, k, &one, b, tb ? k : n, a, ta ? m : k, &zero, c, n));
    }
    double norm(bool rms) {
        double squared = 0.;
        blas_check(cublasDdot(blas, error_stride, ptr(error), 1, ptr(error), 1, &squared));
        counters["scalar_device_to_host_bytes"] += sizeof(double);
        if (!std::isfinite(squared) || squared < 0.)
            throw PSIEXCEPTION("cuEST DIIS: nonfinite or negative error norm.");
        if (rms) return std::sqrt(squared / error_stride);
        int index = 0;
        blas_check(cublasIdamax(blas, error_stride, ptr(error), 1, &index));
        double value = 0.;
        cuda_check(cudaMemcpyAsync(&value, ptr(error) + index-1, sizeof(double), cudaMemcpyDeviceToHost, stream_handle));
        cuda_check(cudaStreamSynchronize(stream_handle));
        counters["scalar_device_to_host_bytes"] += sizeof(int) + sizeof(double);
        return std::abs(value);
    }
    void commit() {
        if (count < cap) pivot = count++;
        else if (oldest) pivot = iteration % cap;
        else {
            pivot = 0;
            for (int i = 1; i < cap; ++i)
                if (e[i*cap+i] > e[pivot*cap+pivot]) pivot = i;
        }
        ++iteration;
        const auto copy = [&](CuestDeviceArray& dst, CuestDeviceArray& src, int stride) {
            cuda_check(cudaMemcpyAsync(ptr(dst) + std::size_t(pivot)*stride, ptr(src),
                std::size_t(stride)*sizeof(double), cudaMemcpyDeviceToDevice, stream_handle));
        };
        copy(history_f, f, ao_stride);
        if (keep_error) copy(history_e, error, error_stride);
        if (keep_density) copy(history_d, d, ao_stride);
        const double one = 1., zero = 0.;
        int offset = 0;
        const auto gram_row = [&](CuestDeviceArray& history, int stride, double* current) {
            blas_check(cublasDgemv(blas, CUBLAS_OP_T, stride, count, &one, ptr(history), stride,
                current, 1, &zero, ptr(small)+offset, 1));
            offset += count;
        };
        if (keep_error) gram_row(history_e, error_stride, ptr(error));
        if (keep_density) {
            gram_row(history_d, ao_stride, ptr(f));  // Q[:,pivot]
            gram_row(history_f, ao_stride, ptr(d));  // Q[pivot,:]
        }
        std::vector<double> values(offset);
        cuda_check(cudaMemcpyAsync(values.data(), ptr(small), values.size()*sizeof(double),
                                  cudaMemcpyDeviceToHost, stream_handle));
        cuda_check(cudaStreamSynchronize(stream_handle));
        counters["scalar_device_to_host_bytes"] += values.size()*sizeof(double);
        if (!std::all_of(values.begin(), values.end(), [](double v) { return std::isfinite(v); }))
            throw PSIEXCEPTION("cuEST DIIS: nonfinite Gram matrix.");
        offset = 0;
        if (keep_error) {
            for (int i = 0; i < count; ++i) e[i*cap+pivot] = e[pivot*cap+i] = values[offset+i];
            offset += count;
        }
        if (keep_density)
            for (int i = 0; i < count; ++i) {
                q[i*cap+pivot] = values[offset+i];
                q[pivot*cap+i] = values[offset+count+i];
            }
        counters["entries_added"] += 1.;
    }
    std::vector<double> gram(const std::vector<double>& data) const {
        if (data.empty()) throw PSIEXCEPTION("cuEST DIIS: requested Gram matrix was not enabled.");
        std::vector<double> result(std::size_t(count)*count);
        for (int i = 0; i < count; ++i)
            for (int j = 0; j < count; ++j) result[i*count+j] = data[i*cap+j];
        return result;
    }
};

CuestDIIS::CuestDIIS(int vectors, int memory_mb, bool errors, bool densities, bool oldest)
    : impl_(std::make_unique<Impl>(vectors, memory_mb, errors, densities, oldest)) {}
CuestDIIS::~CuestDIIS() = default;
double CuestDIIS::scf_update(const std::vector<SharedMatrix>& focks, const std::vector<SharedMatrix>& densities,
                           SharedMatrix overlap, SharedMatrix orthogonalizer, bool save, bool rms) {
    PendingWork pending;
    auto& p = *impl_;
    if (!overlap || !orthogonalizer) throw PSIEXCEPTION("cuEST DIIS requires S and X.");
    const int n = overlap->rowdim(), r = orthogonalizer->coldim();
    check_matrix(overlap, n, n); check_matrix(orthogonalizer, n, r);
    if (focks.empty() || focks.size() > 2 || densities.size() != focks.size())
        throw PSIEXCEPTION("cuEST DIIS: inconsistent spin dimensions.");
    for (const auto& matrix : focks) check_matrix(matrix, n, n);
    for (const auto& matrix : densities) check_matrix(matrix, n, n);
    p.initialize(n, r, focks.size());
    p.upload_spins(focks, p.f, n); p.upload_spins(densities, p.d, n);
    if (!p.fixed_ready) {
        p.upload(overlap, p.ptr(p.s)); p.upload(orthogonalizer, p.ptr(p.x));
        p.counters["fixed_input_uploads"] += 1.;
        p.counters["fixed_input_host_to_device_bytes"] +=
            (std::size_t(n)*n + std::size_t(n)*r)*sizeof(double);
        p.fixed_ready = true;
    }
    const double one = 1., minus_one = -1.;
    for (int spin = 0; spin < p.nspin; ++spin) {
        const auto offset = std::size_t(spin)*n*n;
        p.gemm(false, false, n, n, n, p.ptr(p.f)+offset, p.ptr(p.d)+offset, p.ptr(p.work1));
        p.gemm(false, false, n, n, n, p.ptr(p.work1), p.ptr(p.s), p.ptr(p.work2));
        blas_check(cublasDgeam(p.blas, CUBLAS_OP_N, CUBLAS_OP_T, n, n,
            &one, p.ptr(p.work2), n, &minus_one, p.ptr(p.work2), n, p.ptr(p.work1), n));
        p.gemm(false, false, n, r, n, p.ptr(p.work1), p.ptr(p.x), p.ptr(p.work2));
        p.gemm(true, false, r, r, n, p.ptr(p.x), p.ptr(p.work2),
               p.ptr(p.error)+std::size_t(spin)*r*r);
    }
    const double result = p.norm(rms);
    if (save) p.commit();
    p.counters["gradient_updates"] += 1.;
    pending.done = true;
    return result;
}
void CuestDIIS::add_entry(const std::vector<SharedMatrix>& errors, const std::vector<SharedMatrix>& targets,
                         const std::vector<SharedMatrix>& densities) {
    PendingWork pending;
    auto& p = *impl_;
    if (targets.empty() || !targets[0] || (p.keep_error && (errors.empty() || !errors[0])))
        throw PSIEXCEPTION("cuEST DIIS: missing entry matrices.");
    const int n = targets[0]->rowdim();
    const int r = p.keep_error ? errors[0]->rowdim() : n;
    if (targets.size() > 2 || (p.keep_error && errors.size() != targets.size()) ||
        (p.keep_density && densities.size() != targets.size()))
        throw PSIEXCEPTION("cuEST DIIS: inconsistent entry spin dimensions.");
    for (const auto& matrix : targets) check_matrix(matrix, n, n);
    if (p.keep_error) for (const auto& matrix : errors) check_matrix(matrix, r, r);
    if (p.keep_density) for (const auto& matrix : densities) check_matrix(matrix, n, n);
    p.initialize(n, r, targets.size());
    p.upload_spins(targets, p.f, n);
    if (p.keep_density) p.upload_spins(densities, p.d, n);
    if (p.keep_error) {
        p.upload_spins(errors, p.error, r);
        p.norm(true);  // Reject nonfinite errors before touching the history.
    }
    p.commit();
    pending.done = true;
}
void CuestDIIS::extrapolate(const std::vector<double>& coefficients, const std::vector<SharedMatrix>& targets) {
    PendingWork pending;
    auto& p = *impl_;
    if (!p.count || coefficients.size() != static_cast<std::size_t>(p.count) ||
        targets.size() != static_cast<std::size_t>(p.nspin) ||
        !std::all_of(coefficients.begin(), coefficients.end(), [](double v) { return std::isfinite(v); }))
        throw PSIEXCEPTION("cuEST DIIS: invalid extrapolation shape or coefficients.");
    for (const auto& target : targets) check_matrix(target, p.nbf, p.nbf);
    cuda_check(cudaMemcpyAsync(p.ptr(p.coeff), coefficients.data(), coefficients.size()*sizeof(double),
                              cudaMemcpyHostToDevice, stream_handle));
    p.counters["scalar_host_to_device_bytes"] += coefficients.size()*sizeof(double);
    const double one = 1., zero = 0.;
    blas_check(cublasDgemv(p.blas, CUBLAS_OP_N, p.ao_stride, p.count, &one,
        p.ptr(p.history_f), p.ao_stride, p.ptr(p.coeff), 1, &zero, p.ptr(p.f), 1));
    const auto bytes = std::size_t(p.nbf)*p.nbf*sizeof(double);
    for (int spin = 0; spin < p.nspin; ++spin)
        cuda_check(cudaMemcpyAsync(targets[spin]->pointer()[0],
            p.ptr(p.f)+std::size_t(spin)*p.nbf*p.nbf, bytes, cudaMemcpyDeviceToHost, stream_handle));
    cuda_check(cudaStreamSynchronize(stream_handle));
    p.counters["matrix_device_to_host_bytes"] += bytes*p.nspin;
    p.counters["extrapolations"] += 1.;
    pending.done = true;
}
std::vector<double> CuestDIIS::error_gram() const {
    if (!impl_->keep_error) throw PSIEXCEPTION("cuEST DIIS: error history was not enabled.");
    return impl_->gram(impl_->e);
}
std::vector<double> CuestDIIS::density_fock_gram() const { return impl_->gram(impl_->q); }
std::map<std::string, double> CuestDIIS::stats() const { return impl_->counters; }
int CuestDIIS::size() const { return impl_->count; }
int CuestDIIS::capacity() const { return impl_->cap; }
int CuestDIIS::last_pivot() const { return impl_->pivot; }
void CuestDIIS::reset_subspace() { impl_->release(); impl_->counters["subspace_resets"] += 1.; }
void CuestDIIS::clear() { impl_->release(); }
}  // namespace psi
#endif

/* SPDX-License-Identifier: LGPL-3.0-only
 *
 * Independent, device-resident PCG. This does not call the host cphf_Hx or
 * JK::compute adapters. Psi4's H has negative orbital-gap sign: solve
 * (-H)x=-b using positive pseudodensities and a positive-gap preconditioner.
 * Numerical updates/reductions stay on device; the host only schedules work
 * and reads one scalar residual/status summary per equation.
 */
#include "hf.h"
#include "psi4/liboptions/liboptions.h"
#include "psi4/libpsi4util/exception.h"
#include "psi4/libpsi4util/PsiOutStream.h"
#include "psi4/libmints/matrix.h"
#include <algorithm>
#include <cmath>
#include <limits>

#ifdef PSI4_CUEST_DEVICE_CG
#include "cpscf_exchange.h"
#include "psi4/libfock/cuESTCommon.h"
#include "psi4/libfock/cuest_device_response.h"
#include "psi4/libfock/cuest_device_kernels.h"
#include "psi4/libfock/cuest_response_profile.h"
#include "psi4/libfock/v.h"
#include "psi4/libfunctional/superfunctional.h"
#include "psi4/libmints/basisset.h"
#include "psi4/libqt/qt.h"
#include <array>
#include <chrono>
#include <cublas_v2.h>
#include <cuda_runtime.h>
extern cudaStream_t stream_handle;
#endif

namespace psi {
namespace scf {
#ifdef PSI4_CUEST_DEVICE_CG
namespace {
using Clock = std::chrono::steady_clock;
double seconds(Clock::time_point a) { return std::chrono::duration<double>(Clock::now() - a).count(); }
void cg_cuda(cudaError_t s) {
    if (s != cudaSuccess) throw PSIEXCEPTION(std::string("cuEST device CG CUDA error: ") + cudaGetErrorString(s));
}
void cg_blas(cublasStatus_t s) {
    if (s != CUBLAS_STATUS_SUCCESS)
        throw PSIEXCEPTION("cuEST device CG cuBLAS error " + std::to_string(static_cast<int>(s)));
}
std::size_t product(std::size_t a, std::size_t b) {
    if (a && b > std::numeric_limits<std::size_t>::max() / a)
        throw PSIEXCEPTION("cuEST device CG: buffer size overflow.");
    return a * b;
}
double* allocate(CuestDeviceArray& a, std::size_t n) {
    product(n, sizeof(double));
    return a.ensure(n, "device CG");
}
struct Blas {
    cublasHandle_t handle = nullptr;
    Blas() {
        cg_blas(cublasCreate(&handle));
        try { cg_blas(cublasSetStream(handle, stream_handle)); }
        catch (...) { cublasDestroy(handle); handle = nullptr; throw; }
    }
    ~Blas() { if (handle) cublasDestroy(handle); }
    // Row-major batched GEMM, reversing operands for column-major cuBLAS.
    void gemm(bool ta, bool tb, int m, int n, int k, double alpha,
              const double* a, long long sa, const double* b, long long sb,
              double beta, double* c, long long sc, int batch) {
        if (!m || !n || !batch) return;
        cg_blas(cublasDgemmStridedBatched(handle, tb ? CUBLAS_OP_T : CUBLAS_OP_N,
            ta ? CUBLAS_OP_T : CUBLAS_OP_N, n, m, k, &alpha,
            b, tb ? k : n, sb, a, ta ? m : k, sa, &beta, c, n, sc, batch));
    }
    void dots(std::size_t nrhs, int stride, const double* x, const double* y, double* out) {
        cg_blas(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE));
        try {
            for (std::size_t r = 0; r < nrhs; ++r)
                cg_blas(cublasDdot(handle, stride, x + r * stride, 1, y + r * stride, 1, out + r));
        } catch (...) {
            cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST);
            throw;
        }
        cg_blas(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));
    }
};
struct Spin {
    std::size_t no = 0, nv = 0, offset = 0;
    CuestDeviceArray co, cv, foo, fvv, right, projected;
    double *dco = nullptr, *dcv = nullptr, *dfoo = nullptr, *dfvv = nullptr;
    double *dr = nullptr, *dp = nullptr;
};

/// Full Fock-panel operator with bounded AO/grid tiles, persistent device buffers
/// and fixed-reference XC. All RHS MO vectors remain resident even for large batches.
class DeviceHessian {
    Blas& blas_;
    CuestDeviceKernels& kernels_;
    std::array<Spin, 2> spins_;
    std::size_t nbf_, n2_, nrhs_, nspin_, stride_, tile_;
    bool exchange_, xc_, profile_;
    double kscale_;
    CuestDeviceArray density_, coulomb_, exchange_data_, potential_, fock_;
    double *dD_, *dJ_, *dK_, *dV_, *dG_;
    std::unique_ptr<CuestJKDeviceResponse> jk_;
    std::unique_ptr<CuestXCDeviceResponse> vx_;
    std::map<std::string, double>& stats_;
   public:
    DeviceHessian(Blas& blas, cuESTJK& jk, VBase* potential, std::size_t nbf,
                  const std::vector<SharedMatrix>& co, const std::vector<SharedMatrix>& cv,
                  const std::vector<SharedMatrix>& foo, const std::vector<SharedMatrix>& fvv,
                  std::size_t nrhs, std::size_t stride, bool use_k, double kscale,
                  std::map<std::string, double>& stats, bool profile)
        : blas_(blas), kernels_(CuestDeviceKernels::instance()), nbf_(nbf), n2_(product(nbf, nbf)),
          nrhs_(nrhs), nspin_(co.size()), stride_(stride), exchange_(use_k),
          xc_(potential != nullptr), profile_(profile), kscale_(kscale), stats_(stats) {
        const auto per_rhs = product(product(n2_, nspin_), 6 * sizeof(double));
        tile_ = std::min(nrhs_, std::max<std::size_t>(1, std::min<std::size_t>(8, (std::size_t(256) << 20) /
                                                                      std::max<std::size_t>(1, per_rhs))));
        std::vector<std::size_t> nocc;
        std::size_t offset = 0;
        const auto upload = [&](CuestDeviceArray& b, const SharedMatrix& m) {
            const std::size_t count = product(m->rowdim(), m->coldim());
            double* d = allocate(b, count);
            if (count) cg_cuda(cudaMemcpyAsync(d, m->pointer()[0], count * sizeof(double),
                                               cudaMemcpyHostToDevice, stream_handle));
            stats_["setup_host_to_device_bytes"] += count * sizeof(double);
            // Pageable source matrices may be temporary (transposes).
            cg_cuda(cudaStreamSynchronize(stream_handle));
            return d;
        };
        CuestDeviceJKView view;
        view.nrhs = tile_;
        view.compute_exchange = exchange_;
        view.exchange_sum_only = true;
        for (std::size_t s = 0; s < nspin_; ++s) {
            auto& t = spins_[s];
            t.no = co[s]->coldim(); t.nv = cv[s]->coldim(); t.offset = offset;
            offset += product(t.no, t.nv);
            t.dco = upload(t.co, co[s]->transpose());
            t.dcv = upload(t.cv, cv[s]->transpose());
            t.dfoo = upload(t.foo, foo[s]); t.dfvv = upload(t.fvv, fvv[s]);
            t.dr = allocate(t.right, product(tile_, product(t.no, nbf_)));
            t.dp = allocate(t.projected, product(tile_, product(t.no, nbf_)));
            nocc.push_back(t.no);
            view.left[s] = t.dco; view.right[s] = t.dr;
        }
        const auto ao_count = product(product(tile_, nspin_), n2_);
        dD_ = allocate(density_, ao_count); dJ_ = allocate(coulomb_, ao_count);
        dK_ = allocate(exchange_data_, exchange_ ? ao_count : 0);
        dV_ = allocate(potential_, xc_ ? ao_count : 0); dG_ = allocate(fock_, ao_count);
        view.nocc = nocc; view.density = dD_; view.coulomb = dJ_; view.exchange = dK_;
        jk_ = std::make_unique<CuestJKDeviceResponse>(jk, view);
        stats_["jk_workspace_device_bytes"] = jk_->device_workspace_bytes();
        stats_["jk_density_batch_size"] = jk_->density_batch_size();
        stats_["symmetric_gridk"] = jk_->symmetric_gridk();
        stats_["ao_rhs_tile"] = tile_;
        if (xc_) {
            vx_ = potential->prepare_device_response(nocc);
            std::array<double*, 2> out{dV_, nspin_ == 2 ? dV_ + tile_ * n2_ : nullptr};
            vx_->prepare(view.left, view.right, out);
        }
    }
    void apply(const double* x, double* ax) {
        CuestResponseProfile total(profile_, "cuEST response: Hx");
        for (std::size_t start = 0; start < nrhs_; start += tile_) {
            const auto count = std::min(tile_, nrhs_ - start);
            CuestResponseProfile factors(profile_, "cuEST response: factors");
            for (std::size_t s = 0; s < nspin_; ++s) {
                auto& t = spins_[s];
                double* d = dD_ + s * tile_ * n2_;
                const auto right_stride = t.no * nbf_;
                if (!t.no || !t.nv) {
                    cg_cuda(cudaMemsetAsync(d, 0, tile_ * n2_ * sizeof(double), stream_handle));
                    if (right_stride)
                        cg_cuda(cudaMemsetAsync(t.dr, 0, tile_ * right_stride * sizeof(double), stream_handle));
                    continue;
                }
                blas_.gemm(false, false, t.no, nbf_, t.nv, 1.,
                    x + start * stride_ + t.offset, stride_, t.dcv, 0, 0., t.dr, right_stride, count);
                blas_.gemm(true, false, nbf_, nbf_, t.no, 1.,
                    t.dco, 0, t.dr, right_stride, 0., d, n2_, count);
                if (count < tile_) {
                    cg_cuda(cudaMemsetAsync(d + count * n2_, 0, (tile_ - count) * n2_ * sizeof(double), stream_handle));
                    cg_cuda(cudaMemsetAsync(t.dr + count * right_stride, 0,
                        (tile_ - count) * right_stride * sizeof(double), stream_handle));
                }
            }
            factors.stop();
            jk_->compute();
            if (xc_) {
                for (std::size_t r = 0; r < count; ++r) {
                    std::array<const double*, 2> left{}, right{};
                    std::array<double*, 2> out{};
                    for (std::size_t s = 0; s < nspin_; ++s) {
                        left[s] = spins_[s].dco;
                        right[s] = spins_[s].no ? spins_[s].dr + r * spins_[s].no * nbf_ : nullptr;
                        out[s] = dV_ + (s * tile_ + r) * n2_;
                    }
                    vx_->apply(left, right, out);
                }
                if (count < tile_)
                    for (std::size_t s = 0; s < nspin_; ++s)
                        cg_cuda(cudaMemsetAsync(dV_ + (s * tile_ + count) * n2_, 0,
                            (tile_ - count) * n2_ * sizeof(double), stream_handle));
            }
            CuestResponseProfile projection(profile_, "cuEST response: projection");
            kernels_.launch("cg_assemble", nspin_ * tile_ * n2_,
                nbf_, tile_, nspin_, int(exchange_), int(xc_), kscale_, dJ_, dK_, dV_, dG_);
            for (std::size_t s = 0; s < nspin_; ++s) {
                auto& t = spins_[s];
                if (!t.no || !t.nv) continue;
                const double* input = x + start * stride_ + t.offset;
                double* output = ax + start * stride_ + t.offset;
                blas_.gemm(false, false, t.no, nbf_, nbf_, 1., t.dco, 0,
                    dG_ + s * tile_ * n2_, n2_, 0., t.dp, t.no * nbf_, count);
                blas_.gemm(false, true, t.no, t.nv, nbf_, 1., t.dp, t.no * nbf_,
                    t.dcv, 0, 0., output, stride_, count);
                // Full current Fock panels, including noncanonical references.
                blas_.gemm(false, false, t.no, t.nv, t.nv, 1., input, stride_,
                    t.dfvv, 0, 1., output, stride_, count);
                blas_.gemm(false, false, t.no, t.nv, t.no, -1., t.dfoo, 0,
                    input, stride_, 1., output, stride_, count);
            }
        }
        stats_["operator_calls"] += 1;
    }
};
}  // namespace
#endif

std::vector<SharedMatrix> HF::cphf_solve_device(const std::vector<SharedMatrix>& rhs, double tolerance,
                                               int max_iter, int print_lvl, bool unrestricted) {
    cphf_converged_ = false;
    cphf_nfock_builds_ = 0;
    cphf_device_stats_.clear();
#ifndef PSI4_CUEST_DEVICE_CG
    throw PSIEXCEPTION("CUEST_DEVICE_CG requires a Psi4 build with cuEST, CUDA NVRTC and CUDA driver support.");
#else
    const auto total_start = Clock::now();
    if (!std::isfinite(tolerance) || tolerance <= 0 || max_iter < 0)
        throw PSIEXCEPTION("CUEST_DEVICE_CG requires a positive finite tolerance and nonnegative iteration limit.");
    if (nirrep_ != 1) throw PSIEXCEPTION("CUEST_DEVICE_CG currently requires C1 molecular symmetry.");
    const auto ref = options_.get_str("REFERENCE");
    if (ref != "RHF" && ref != "RKS" && ref != "UHF" && ref != "UKS")
        throw PSIEXCEPTION("CUEST_DEVICE_CG supports only RHF/RKS and UHF/UKS references.");
    if (frac_performed_) throw PSIEXCEPTION("CUEST_DEVICE_CG does not support fractional occupations.");
    if (functional_->needs_vv10() || functional_->ansatz() >= 2)
        throw PSIEXCEPTION("CUEST_DEVICE_CG does not support MGGA or VV10 response kernels.");
    if (!external_cpscf_perturbations_.empty() || PCM_enabled() ||
        options_.get_bool("DDX") || options_.get_bool("PE"))
        throw PSIEXCEPTION("CUEST_DEVICE_CG does not support external/PCM/DDX/PE response contributions.");
    auto jk = std::dynamic_pointer_cast<cuESTJK>(jk_);
    if (!jk) throw PSIEXCEPTION("CUEST_DEVICE_CG requires an attached cuESTJK builder (USE_CUEST, SAVE_JK).");
    if (!jk->has_initialized_plan())
        throw PSIEXCEPTION("CUEST_DEVICE_CG requires an initialized JK plan; initialize after finalize/failure.");
    const std::size_t nspin = unrestricted ? 2 : 1;
    if (rhs.size() % nspin) throw PSIEXCEPTION("CUEST_DEVICE_CG expects alternating alpha/beta RHS pairs.");
    const std::size_t nrhs = rhs.size() / nspin;
    auto& stats = cphf_device_stats_;
    for (const auto* key : {"setup_host_to_device_bytes", "result_device_to_host_bytes",
                            "iteration_bulk_host_to_device_bytes", "iteration_bulk_device_to_host_bytes",
                            "iteration_scalar_device_to_host_bytes", "operator_calls", "cg_updates"})
        stats[key] = 0.;
    stats["rhs_count"] = nrhs; stats["spin_count"] = nspin;
    std::vector<SharedMatrix> ret;
    if (!nrhs) { cphf_converged_ = true; return ret; }
    std::vector<SharedMatrix> co{Ca_subset("SO", "OCC")}, cv{Ca_subset("SO", "VIR")}, f{Fa_};
    if (unrestricted) { co.push_back(Cb_subset("SO", "OCC")); cv.push_back(Cb_subset("SO", "VIR")); f.push_back(Fb_); }
    std::vector<SharedMatrix> foo, fvv;
    std::vector<double> gaps;
    std::array<std::size_t, 2> offsets{};
    std::size_t stride = 0;
    for (std::size_t s = 0; s < nspin; ++s) {
        offsets[s] = stride;
        if (co[s]->rowdim() != basisset()->nbf() || cv[s]->rowdim() != basisset()->nbf())
            throw PSIEXCEPTION("CUEST_DEVICE_CG requires dense AO-compatible C1 orbital coefficients.");
        foo.push_back(linalg::triplet(co[s], f[s], co[s], true, false, false));
        fvv.push_back(linalg::triplet(cv[s], f[s], cv[s], true, false, false));
        for (int i = 0; i < co[s]->coldim(); ++i)
            for (int a = 0; a < cv[s]->coldim(); ++a) {
                const double gap = fvv[s]->get(a, a) - foo[s]->get(i, i);
                if (!std::isfinite(gap) || gap <= 1.e-12)
                    throw PSIEXCEPTION("CUEST_DEVICE_CG requires a finite positive orbital-gap preconditioner.");
                gaps.push_back(gap);
            }
        stride += product(co[s]->coldim(), cv[s]->coldim());
    }
    if (stride > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        throw PSIEXCEPTION("CUEST_DEVICE_CG equation exceeds cuBLAS vector size limits.");
    const auto n = product(nrhs, stride);
    std::vector<double> host_b(n);
    bool nonzero = false;
    for (std::size_t r = 0; r < nrhs; ++r)
        for (std::size_t s = 0; s < nspin; ++s) {
            const auto& b = rhs[r * nspin + s];
            if (!b || b->nirrep() != 1 || b->symmetry() != 0 ||
                b->rowdim() != co[s]->coldim() || b->coldim() != cv[s]->coldim())
                throw PSIEXCEPTION("CUEST_DEVICE_CG RHS must be dense C1 occupied x virtual matrices.");
            auto out = b->clone(); out->zero(); ret.push_back(out);
            const auto count = product(b->rowdim(), b->coldim());
            for (std::size_t i = 0; i < count; ++i) {
                const double value = b->pointer()[0][i];
                if (!std::isfinite(value)) throw PSIEXCEPTION("CUEST_DEVICE_CG RHS contains nonfinite values.");
                host_b[r * stride + offsets[s] + i] = -value;
                nonzero = nonzero || value != 0.;
            }
        }
    if (!stride || !nonzero) {
        cphf_converged_ = true;
        stats["final_max_relative_residual"] = 0.;
        stats["total_seconds"] = seconds(total_start);
        return ret;
    }
    const bool use_k = functional_->is_x_hybrid() || functional_->is_x_lrc();
    const double kscale = use_k ? cpscf::exchange_scale(*jk, *functional_) : 0.;
    auto potential = functional_->needs_xc() ? V_potential() : nullptr;
    if (functional_->needs_xc() && !potential)
        throw PSIEXCEPTION("CUEST_DEVICE_CG is missing its DFT response potential.");
    cuest_common::ensure_cuest_initialized();
    Blas blas;
    auto& kernels = CuestDeviceKernels::instance();
    CuestDeviceArray bbuf, xbuf, rbuf, zbuf, pbuf, apbuf, gapbuf;
    CuestDeviceArray normbuf, rrbuf, oldbuf, newbuf, sigmabuf, metricbuf, activebuf, errorbuf;
    double *b = allocate(bbuf, n), *x = allocate(xbuf, n), *r = allocate(rbuf, n);
    double *z = allocate(zbuf, n), *p = allocate(pbuf, n), *ap = allocate(apbuf, n);
    double *gap = allocate(gapbuf, stride), *bn = allocate(normbuf, nrhs), *rr = allocate(rrbuf, nrhs);
    double *oldrho = allocate(oldbuf, nrhs), *newrho = allocate(newbuf, nrhs);
    double *sigma = allocate(sigmabuf, nrhs), *metric = allocate(metricbuf, nrhs);
    auto active = static_cast<int*>(activebuf.ensure_bytes(nrhs * sizeof(int), "CG active mask"));
    auto error = static_cast<int*>(errorbuf.ensure_bytes(sizeof(int), "CG status"));
    cg_cuda(cudaMemcpyAsync(b, host_b.data(), n * sizeof(double), cudaMemcpyHostToDevice, stream_handle));
    cg_cuda(cudaMemcpyAsync(gap, gaps.data(), stride * sizeof(double), cudaMemcpyHostToDevice, stream_handle));
    stats["setup_host_to_device_bytes"] += (n + stride) * sizeof(double);
    cg_cuda(cudaMemsetAsync(error, 0, sizeof(int), stream_handle));
    blas.dots(nrhs, stride, b, b, bn);
    kernels.launch("cg_init", n, n, stride, tolerance * tolerance, gap, b, bn, x, r, z, p, active, error);
    blas.dots(nrhs, stride, r, z, oldrho);
    blas.dots(nrhs, stride, r, r, rr);
    std::vector<double> residuals(nrhs);
    const auto monitor = [&](bool iteration) {
        kernels.launch("cg_metrics", nrhs, nrhs, tolerance * tolerance, rr, bn, metric, active, error);
        int status = 0;
        cg_cuda(cudaMemcpyAsync(residuals.data(), metric, nrhs * sizeof(double), cudaMemcpyDeviceToHost, stream_handle));
        cg_cuda(cudaMemcpyAsync(&status, error, sizeof(int), cudaMemcpyDeviceToHost, stream_handle));
        cg_cuda(cudaStreamSynchronize(stream_handle));
        if (iteration) stats["iteration_scalar_device_to_host_bytes"] += nrhs * sizeof(double) + sizeof(int);
        if (status) throw PSIEXCEPTION("cuEST device CG breakdown/nonfinite or nonpositive curvature (status " +
                                      std::to_string(status) + "). Check SCF stability and preconditioning.");
        double max_value = 0.;
        for (double value : residuals) max_value = std::max(max_value, std::sqrt(value));
        return max_value;
    };
    double max_residual = monitor(false);
    if (max_residual <= tolerance) {
        cphf_converged_ = true;
        stats["final_max_relative_residual"] = max_residual;
        stats["total_seconds"] = seconds(total_start);
        return ret;
    }
    const bool profile = options_.get_bool("CUEST_RESPONSE_PROFILE");
    DeviceHessian hessian(blas, *jk, potential.get(), basisset()->nbf(), co, cv, foo, fvv,
                           nrhs, stride, use_k, kscale, stats, profile);
    cg_cuda(cudaStreamSynchronize(stream_handle));
    stats["setup_seconds"] = seconds(total_start);
    if (print_lvl) outfile->Printf("\n    ==> cuEST device-resident CG (%zu equations) <==\n", nrhs);
    const auto iteration_start = Clock::now();
    bool certified = false;
    const auto certify = [&]() {
        hessian.apply(x, ap);
        kernels.launch("cg_true_residual", n, n, stride, b, ap, gap, r, z);
        blas.dots(nrhs, stride, r, r, rr);
        blas.dots(nrhs, stride, r, z, newrho);
        max_residual = monitor(true);
        kernels.launch("cg_direction", n, n, stride, 1, z, p, oldrho, newrho, active, error);
        std::swap(oldrho, newrho);
        return max_residual <= tolerance;
    };
    for (int iter = 0; iter < max_iter; ++iter) {
        hessian.apply(p, ap);
        CuestResponseProfile updates(profile, "cuEST response: CG updates");
        blas.dots(nrhs, stride, p, ap, sigma);
        kernels.launch("cg_step", n, n, stride, x, r, p, ap, oldrho, sigma, active, error);
        kernels.launch("cg_precondition", n, n, stride, gap, r, z);
        blas.dots(nrhs, stride, r, r, rr);
        blas.dots(nrhs, stride, r, z, newrho);
        max_residual = monitor(true);
        stats["cg_updates"] = iter + 1;
        kernels.launch("cg_direction", n, n, stride, 0, z, p, oldrho, newrho, active, error);
        std::swap(oldrho, newrho);
        updates.stop();
        if (print_lvl) outfile->Printf("      %4d  max relative residual %12.4e\n", iter + 1, max_residual);
        if (max_residual <= tolerance && (certified = certify())) break;
    }
    if (!certified) certified = certify();
    cg_cuda(cudaStreamSynchronize(stream_handle));
    int final_status = 0;
    cg_cuda(cudaMemcpy(&final_status, error, sizeof(int), cudaMemcpyDeviceToHost));
    stats["iteration_scalar_device_to_host_bytes"] += sizeof(int);
    if (final_status)
        throw PSIEXCEPTION("cuEST device CG: nonfinite/breakdown during true-residual certification.");
    stats["iteration_seconds"] = seconds(iteration_start);
    cphf_converged_ = certified;
    cphf_nfock_builds_ = static_cast<int>(std::min<double>(std::numeric_limits<int>::max(),
                                                        stats["operator_calls"] * nrhs));
    std::vector<double> host_x(n);
    cg_cuda(cudaMemcpyAsync(host_x.data(), x, n * sizeof(double), cudaMemcpyDeviceToHost, stream_handle));
    cg_cuda(cudaStreamSynchronize(stream_handle));
    stats["result_device_to_host_bytes"] = n * sizeof(double);
    for (std::size_t v = 0; v < nrhs; ++v)
        for (std::size_t s = 0; s < nspin; ++s) {
            const auto count = product(co[s]->coldim(), cv[s]->coldim());
            if (count) std::copy_n(host_x.data() + v * stride + offsets[s], count, ret[v*nspin+s]->pointer()[0]);
        }
    stats["final_max_relative_residual"] = max_residual;
    stats["total_seconds"] = seconds(total_start);
    return ret;
#endif
}
}  // namespace scf
}  // namespace psi

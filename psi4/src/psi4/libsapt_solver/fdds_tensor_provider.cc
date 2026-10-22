/*
 * Psi4: an open-source quantum chemistry software package
 * Copyright (c) 2026 The Psi4 Developers.
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */
#include "fdds_tensor_provider.h"

#include "psi4/lib3index/dfhelper.h"
#include "psi4/libmints/basisset.h"
#include "psi4/libmints/matrix.h"

#include <climits>
#include <vector>

namespace psi {
namespace sapt {
namespace {
size_t matrix_capacity(const SharedMatrix& matrix) {
    if (!matrix || matrix->nirrep() != 1)
        throw std::runtime_error("FDDS tensor provider: expected a dense one-irrep buffer");
    return FDDSTensorProvider::elements(
        {static_cast<size_t>(matrix->nrow()), static_cast<size_t>(matrix->ncol()), 1});
}

class NativeFDDSTensors final : public FDDSTensorProvider {
   public:
    explicit NativeFDDSTensors(const FDDSTensorPreparation& config)
        : helper_(std::make_unique<DFHelper>(config.primary, config.auxiliary)) {
        helper_->set_memory(config.memory_doubles);
#ifdef USING_cuEST
        helper_->set_metric_gpu(config.metric_gpu, config.metric_gpu_min_dim);
#endif
        helper_->set_nthreads(config.threads);
        helper_->set_method(config.hybrid ? "DIRECT" : "DIRECT_iaQ");
        helper_->set_metric_pow(0.0);
        helper_->initialize();
        helper_->print_header();
        add_spaces(config);
        if (!config.empty[0]) transform("arQ", "a", "r", "pqQ");
        if (!config.empty[1]) transform("bsQ", "b", "s", "pqQ");
        if (config.hybrid) {
            if (!config.empty[0]) {
                transform("raQ", "r", "a", "pqQ");
                transform("Qar", "a", "r", "Qpq");
            }
            if (!config.empty[1]) {
                transform("sbQ", "s", "b", "pqQ");
                transform("Qbs", "b", "s", "Qpq");
            }
        }
        helper_->set_release_core_AO_before_metric(true);
        // Sticky: retain no duplicate pre-metric tensors in either round.
        helper_->set_release_pre_metric_tensors(true);
        helper_->transform();
        if (config.hybrid) {
            // Raw outputs survive this reset; only registrations are cleared.
            helper_->clear_spaces();
            helper_->clear_transformations();
            helper_->set_method("DIRECT_iaQ");
            helper_->set_metric_pow(-0.5);
            current_kind_ = FDDSTensorKind::MetricInverseHalf;
            helper_->initialize();
            add_spaces(config);
            if (!config.empty[0]) {
                transform("aaR", "a", "a", "pqQ");
                transform("arR", "a", "r", "pqQ");
                transform("rrR", "r", "r", "pqQ");
            }
            if (!config.empty[1]) {
                transform("bbR", "b", "b", "pqQ");
                transform("bsR", "b", "s", "pqQ");
                transform("ssR", "s", "s", "pqQ");
            }
            helper_->set_release_core_AO_before_metric(true);
            helper_->transform();
        }
        helper_->clear_spaces();
        // Query the persistent storage registry, not the cleared registrations.
        for (const auto& item : produced_) {
            const auto dimensions = helper_->get_tensor_shape(item.first);
            register_integral(item.first, {std::get<0>(dimensions), std::get<1>(dimensions), std::get<2>(dimensions)},
                              item.second);
        }
    }

   private:
    std::unique_ptr<DFHelper> helper_;
    std::vector<std::pair<std::string, FDDSTensorKind>> produced_;
    FDDSTensorKind current_kind_ = FDDSTensorKind::RawCoulomb;
    void add_spaces(const FDDSTensorPreparation& config) {
        if (!config.empty[0]) {
            helper_->add_space("a", config.coefficients[0]);
            helper_->add_space("r", config.coefficients[1]);
        }
        if (!config.empty[1]) {
            helper_->add_space("b", config.coefficients[2]);
            helper_->add_space("s", config.coefficients[3]);
        }
    }
    void transform(const std::string& name, const std::string& left, const std::string& right,
                   const std::string& layout) {
        helper_->add_transformation(name, left, right, layout);
        // Only these two preparation rounds exist. Names are retained for
        // compatibility; the operator meaning is recorded independently.
        produced_.emplace_back(name, current_kind_);
    }
    void read_impl(const std::string& name, double* destination, FDDSTensorSlice slice) override {
        helper_->fill_tensor(name, destination, {slice[0], slice[1]});
    }
    void write_impl(const std::string& name, const double* source, FDDSTensorSlice slice) override {
        // DFHelper's legacy writer takes a mutable pointer but only reads it.
        helper_->write_disk_tensor(name, const_cast<double*>(source), {slice[0], slice[1]});
    }
    void create_impl(const std::string& name, FDDSTensorShape dimensions) override {
        helper_->add_disk_tensor(name, std::make_tuple(dimensions[0], dimensions[1], dimensions[2]));
    }
};
}  // namespace

void FDDSTensorProvider::read(const std::string& name, SharedMatrix destination, FDDSTensorSlice slice) {
    slice_elements(name, slice);
    const auto dimensions = shape(name);
    const auto count0 = slice[1] - slice[0];
    if (count0 > INT_MAX || dimensions[1] > INT_MAX || dimensions[2] > INT_MAX)
        throw std::runtime_error("FDDS tensor provider: dimensions exceed the NumPy Matrix layout");
    const auto capacity = matrix_capacity(destination);
    read(name, capacity ? destination->pointer()[0] : nullptr, capacity, slice);
    // Preserve DFHelper's exported three-dimensional NumPy view contract.
    destination->set_numpy_shape(
        {static_cast<int>(count0), static_cast<int>(dimensions[1]), static_cast<int>(dimensions[2])});
}

void FDDSTensorProvider::write(const std::string& name, SharedMatrix source, FDDSTensorSlice slice) {
    const auto capacity = matrix_capacity(source);
    write(name, capacity ? source->pointer()[0] : nullptr, capacity, slice);
}

std::unique_ptr<FDDSTensorProvider> make_native_fdds_tensors(const FDDSTensorPreparation& config) {
    if (!config.primary || !config.auxiliary || !config.threads || config.threads > INT_MAX)
        throw std::runtime_error("FDDS tensor provider: invalid basis or thread configuration");
    for (size_t monomer = 0; monomer != 2; ++monomer)
        if (!config.empty[monomer] && (!config.coefficients[2 * monomer] || !config.coefficients[2 * monomer + 1]))
            throw std::runtime_error("FDDS tensor provider: missing active orbital coefficients");
    return std::make_unique<NativeFDDSTensors>(config);
}
}  // namespace sapt
}  // namespace psi

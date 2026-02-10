# DFHelper Dispersion Upgrade: Reusing JK Integrals for SAPT Transformations

## Executive Summary

This document outlines a plan to improve DFHelper's out-of-core performance for SAPT dispersion calculations by reusing the 3-index integrals already computed by the JK object, rather than computing them twice.

---

## Problem Statement

### The Fundamental Issue

SAPT calculations currently compute density-fitted 3-index integrals **twice**:

1. **JK Object** (for electrostatics/exchange): Computes `(Q|mn)` integrals in Q-major format
2. **DFHelper** (for transformations/dispersion): Computes `(p|Qq)` integrals in p-major format

This duplication is wasteful since both represent the same physical integrals, just stored differently.

### Storage Format Incompatibility

**DiskDFJK Storage**: Q-major, triangular-packed
```
File: "(Q|mn) Integrals"
Layout: [Q=0: mn_0, mn_1, ...][Q=1: mn_0, mn_1, ...]...
- Auxiliary index (Q) is outermost
- Only upper triangle stored (m >= n)
- Uses dense packing of significant pairs
- Single contiguous read per Q-block = FAST I/O
```

**DFHelper Storage**: p-major, sparse
```
File: AO integrals
Layout: [p=0: Q0q0, Q0q1, ..., Q1q0, ...][p=1: ...]...
- Primary basis index (p) is outermost  
- Full matrix stored (both p,q and q,p)
- Uses per-row sparse indexing (small_skips_, big_skips_)
- Requires nbf_ separate reads per Q-block = SLOW I/O
```

### Why This Matters

When DFHelper reads integrals for J/K builds (`grab_AO()`), it performs `nbf_` separate disk seeks and reads per Q-block. For large basis sets, this creates severe I/O bottlenecks compared to DiskDFJK's single contiguous read.

The code explicitly acknowledges this (dfhelper.cc line 3057-3058):
> "the strided disk reads for the AOs will result in a definite loss to DiskDFJK in the disk-bound realm"

### Why p-major is Needed for Transformations

The transformation workflow `(pq|Q) -> (ia|Q)` contracts over the primary basis index `p`:

```cpp
// Step 2 of transformation (line 2033):
// (pw)(p|Qb) -> (w|Qb)
C_DGEMM('T', 'N', wsize, block_size * bsize, nbf_, 1.0, 
        Wp, wsize,                    // W matrix [nbf_ x wsize]
        Tp, block_size * bsize,       // T buffer [nbf_ x (Q x b)]
        0.0, Fp, block_size * bsize);
```

This DGEMM requires data contiguous over `p` - exactly what p-major storage provides. Converting to Q-major for transformations would require `block_size` separate small DGEMMs instead of one large one, plus loss of sparsity exploitation.

---

## Options Considered

### Option A: Add Q-major Storage to DFHelper (Not Chosen)
- DFHelper writes both p-major and Q-major files
- **Pro**: Clean separation, optimal for both use cases
- **Con**: Doubles disk usage, doesn't leverage existing JK integrals

### Option B: DFHelper Transpose p-major to Q-major (Not Chosen)
- DFHelper transposes its p-major file to Q-major for J/K builds
- **Pro**: Single integral computation
- **Con**: Still computes integrals twice if JK already ran

### Option C: Reuse JK's Q-major Integrals (Chosen)
- Import JK's integrals and transpose to p-major for DFHelper
- **Pro**: Compute integrals only once, leverage JK's efficient storage
- **Con**: Requires interface changes, screening compatibility checks

### Option D: Memory-mapped I/O (Not Chosen)
- Use mmap() instead of explicit reads
- **Pro**: OS handles caching
- **Con**: Doesn't solve fundamental layout mismatch

---

## Chosen Solution: Option C - Reuse JK Integrals

### High-Level Approach

1. JK computes integrals once and saves to disk (Q-major triangular format)
2. DFHelper imports JK's screening information
3. DFHelper performs one-time transpose: Q-major triangular → p-major sparse
4. DFHelper uses transposed integrals for transformations
5. SAPT benefits from single integral computation

### Performance Estimate

For typical SAPT (nbf=500, naux=1500):
- **Current**: 2 full integral computations
- **Proposed**: 1 integral computation + 1 transpose (~5% of integral cost)
- **Net savings**: ~45-50% reduction in integral computation time

---

## Implementation Plan

### Phase 1: Expose JK Integral Information

**Goal**: Make DiskDFJK's integral data accessible to external consumers.

#### Subgoal 1.1: Add Accessor Methods to DiskDFJK

Add public methods to `DiskDFJK` class in `jk.h`:

```cpp
class DiskDFJK : public JK {
public:
    // ... existing code ...
    
    // === New Accessor Methods ===
    
    /// Get the screening function pairs (m,n) with m >= n
    const std::vector<std::pair<int,int>>& function_pairs() const;
    
    /// Get reverse mapping: triangular index -> packed index (-1 if screened)
    const std::vector<long int>& function_pairs_to_dense() const;
    
    /// Get number of significant function pairs
    size_t n_function_pairs() const { return n_function_pairs_; }
    
    /// Get the PSIO unit number for integral file
    size_t integral_unit() const { return unit_; }
    
    /// Get the auxiliary basis set
    std::shared_ptr<BasisSet> auxiliary() const { return auxiliary_; }
    
    /// Check if integrals are currently on disk
    bool integrals_on_disk() const;
};
```

#### Subgoal 1.2: Implement Accessors in DiskDFJK.cc

```cpp
const std::vector<std::pair<int,int>>& DiskDFJK::function_pairs() const {
    if (eri_.empty()) {
        throw PSIEXCEPTION("DiskDFJK::function_pairs(): ERI engines not initialized");
    }
    return eri_.front()->function_pairs();
}

const std::vector<long int>& DiskDFJK::function_pairs_to_dense() const {
    if (eri_.empty()) {
        throw PSIEXCEPTION("DiskDFJK::function_pairs_to_dense(): ERI engines not initialized");
    }
    return eri_.front()->function_pairs_to_dense();
}

bool DiskDFJK::integrals_on_disk() const {
    return (df_ints_io_ == "SAVE" || df_ints_io_ == "LOAD") && 
           psio_->exists(unit_, "(Q|mn) Integrals");
}
```

#### Subgoal 1.3: Add "SAVE" Option Awareness

Ensure `df_ints_io_ = "SAVE"` properly persists integrals:

```cpp
// In DiskDFJK::postiterations()
void DiskDFJK::postiterations() {
    // ... existing cleanup ...
    
    // Don't delete integral file if SAVE was requested
    if (df_ints_io_ != "SAVE") {
        // existing file deletion code
    }
    // else: keep file for later use
}
```

#### Subgoal 1.4: Add Virtual Base Class Method (Optional)

For cleaner polymorphism, add to JK base class:

```cpp
class JK {
public:
    /// Check if this JK type can export its integrals
    virtual bool can_export_integrals() const { return false; }
    
    /// Get integral export information (throws if not supported)
    virtual std::tuple<size_t, size_t, std::shared_ptr<BasisSet>> 
        get_integral_info() const {
        throw PSIEXCEPTION("This JK type does not support integral export");
    }
};
```

**Deliverables for Phase 1:**
- [ ] Accessor methods added to DiskDFJK header
- [ ] Accessor implementations in DiskDFJK.cc
- [ ] SAVE option properly preserves integrals
- [ ] Unit test verifying accessors work correctly

---

### Phase 2: DFHelper Import Interface

**Goal**: Allow DFHelper to receive screening information from external sources.

#### Subgoal 2.1: Add Import Method Declaration

In `dfhelper.h`:

```cpp
class DFHelper {
public:
    // ... existing code ...
    
    // === JK Integration Methods ===
    
    /// Import screening and integral location from a JK object
    /// This allows DFHelper to reuse JK's precomputed integrals
    /// @param function_pairs The (m,n) pairs from JK's Schwarz screening (m >= n)
    /// @param function_pairs_to_dense Reverse mapping for triangular indices
    /// @param n_function_pairs Number of significant pairs
    /// @param jk_unit PSIO unit number where JK stored "(Q|mn) Integrals"
    /// @param auxiliary The auxiliary basis used by JK (must match DFHelper's)
    void import_jk_screening(
        const std::vector<std::pair<int,int>>& function_pairs,
        const std::vector<long int>& function_pairs_to_dense,
        size_t n_function_pairs,
        size_t jk_unit,
        std::shared_ptr<BasisSet> auxiliary
    );
    
    /// Convenience overload that extracts info from DiskDFJK directly
    void import_jk_screening(std::shared_ptr<JK> jk);
    
    /// Check if JK screening has been imported
    bool has_jk_screening() const { return jk_screening_imported_; }
    
    /// Transpose JK's Q-major integrals to DFHelper's p-major format
    /// Must call import_jk_screening() first
    void transpose_jk_integrals();

protected:
    // ... existing code ...
    
    // === JK Integration Members ===
    bool jk_screening_imported_ = false;
    size_t jk_unit_ = 0;
    size_t jk_n_function_pairs_ = 0;
    std::vector<std::pair<int,int>> jk_function_pairs_;
    std::vector<long int> jk_function_pairs_to_dense_;
};
```

#### Subgoal 2.2: Implement Import Methods

In `dfhelper.cc`:

```cpp
void DFHelper::import_jk_screening(
    const std::vector<std::pair<int,int>>& function_pairs,
    const std::vector<long int>& function_pairs_to_dense,
    size_t n_function_pairs,
    size_t jk_unit,
    std::shared_ptr<BasisSet> auxiliary) 
{
    timer_on("DFH: import JK screening");
    
    // Validate auxiliary basis matches
    if (auxiliary->nbf() != naux_) {
        throw PSIEXCEPTION("DFHelper::import_jk_screening(): "
            "Auxiliary basis mismatch. JK has " + std::to_string(auxiliary->nbf()) +
            " functions, DFHelper has " + std::to_string(naux_));
    }
    
    // Store JK's screening info
    jk_function_pairs_ = function_pairs;
    jk_function_pairs_to_dense_ = function_pairs_to_dense;
    jk_n_function_pairs_ = n_function_pairs;
    jk_unit_ = jk_unit;
    
    // Validate screening compatibility
    // DFHelper's screening should be a subset of JK's (or identical)
    validate_screening_compatibility();
    
    jk_screening_imported_ = true;
    
    if (print_lvl_ > 0) {
        outfile->Printf("    DFHelper: Imported JK screening information\n");
        outfile->Printf("      JK function pairs: %zu\n", jk_n_function_pairs_);
        outfile->Printf("      JK integral unit:  %zu\n", jk_unit_);
    }
    
    timer_off("DFH: import JK screening");
}

void DFHelper::import_jk_screening(std::shared_ptr<JK> jk) {
    // Attempt to cast to DiskDFJK
    auto disk_jk = std::dynamic_pointer_cast<DiskDFJK>(jk);
    if (!disk_jk) {
        throw PSIEXCEPTION("DFHelper::import_jk_screening(): "
            "JK object is not a DiskDFJK. Only DiskDFJK supports integral export.");
    }
    
    if (!disk_jk->integrals_on_disk()) {
        throw PSIEXCEPTION("DFHelper::import_jk_screening(): "
            "JK integrals not on disk. Set df_ints_io='SAVE' before JK initialization.");
    }
    
    import_jk_screening(
        disk_jk->function_pairs(),
        disk_jk->function_pairs_to_dense(),
        disk_jk->n_function_pairs(),
        disk_jk->integral_unit(),
        disk_jk->auxiliary()
    );
}
```

#### Subgoal 2.3: Implement Screening Validation

```cpp
void DFHelper::validate_screening_compatibility() {
    // Ensure DFHelper's prepare_sparsity() has been called
    if (!sparsity_prepared_) {
        prepare_sparsity();
    }
    
    size_t missing_in_jk = 0;
    size_t extra_in_jk = 0;
    
    // Check each pair in DFHelper's screening
    for (size_t p = 0; p < nbf_; p++) {
        for (size_t q = 0; q < nbf_; q++) {
            bool dfh_significant = (schwarz_fun_index_[p * nbf_ + q] > 0);
            
            // Map to JK's triangular indexing
            size_t m = std::max(p, q);
            size_t n = std::min(p, q);
            size_t tri_idx = m * (m + 1) / 2 + n;
            
            bool jk_significant = (tri_idx < jk_function_pairs_to_dense_.size() &&
                                   jk_function_pairs_to_dense_[tri_idx] >= 0);
            
            if (dfh_significant && !jk_significant) {
                missing_in_jk++;
            } else if (!dfh_significant && jk_significant) {
                extra_in_jk++;
            }
        }
    }
    
    if (missing_in_jk > 0) {
        outfile->Printf("  WARNING: %zu function pairs in DFHelper screening "
                       "but not in JK screening.\n", missing_in_jk);
        outfile->Printf("           These integrals will be zero in transformations.\n");
        outfile->Printf("           Consider using consistent screening thresholds.\n");
    }
    
    if (print_lvl_ > 1 && extra_in_jk > 0) {
        outfile->Printf("    Note: JK has %zu extra pairs not in DFHelper screening "
                       "(this is fine).\n", extra_in_jk);
    }
}
```

**Deliverables for Phase 2:**
- [ ] Import method declarations in dfhelper.h
- [ ] Import method implementations in dfhelper.cc
- [ ] Screening validation logic
- [ ] Appropriate warnings for screening mismatches
- [ ] Unit test for import functionality

---

### Phase 3: Transpose Implementation

**Goal**: Implement efficient transpose from JK's Q-major triangular format to DFHelper's p-major sparse format.

#### Subgoal 3.1: Implement Core Transpose Logic

```cpp
void DFHelper::transpose_jk_integrals() {
    timer_on("DFH: transpose JK integrals");
    
    if (!jk_screening_imported_) {
        throw PSIEXCEPTION("DFHelper::transpose_jk_integrals(): "
            "Must call import_jk_screening() first");
    }
    
    if (print_lvl_ > 0) {
        outfile->Printf("\n    DFHelper: Transposing JK integrals to p-major format\n");
    }
    
    // Ensure sparsity info is ready
    if (!sparsity_prepared_) {
        prepare_sparsity();
    }
    
    // Open JK's integral file
    auto psio = PSIO::shared_object();
    psio->open(jk_unit_, PSIO_OPEN_OLD);
    
    // Determine memory-based blocking
    // Need: Q-block read buffer + p-block write buffer
    size_t total_pq_sparse = big_skips_[nbf_];  // DFHelper's total sparse size
    
    // Q-block size limited by memory
    // Read buffer: Qblock * jk_n_function_pairs_
    // Write accumulator: we process p-by-p, so just Qblock * max(small_skips_)
    size_t max_small_skip = *std::max_element(small_skips_.begin(), small_skips_.end() - 1);
    size_t mem_per_Q = jk_n_function_pairs_ + max_small_skip;
    size_t Qblock = std::min(naux_, memory_ / (2 * mem_per_Q));
    Qblock = std::max(Qblock, (size_t)1);
    
    if (print_lvl_ > 1) {
        outfile->Printf("      Q-block size: %zu\n", Qblock);
        outfile->Printf("      JK pairs: %zu, DFH sparse size: %zu\n", 
                       jk_n_function_pairs_, total_pq_sparse);
    }
    
    // Allocate buffers
    std::unique_ptr<double[]> Qbuf(new double[Qblock * jk_n_function_pairs_]);
    std::unique_ptr<double[]> pbuf(new double[Qblock * max_small_skip]);
    double* Qp = Qbuf.get();
    double* pp = pbuf.get();
    
    // Create output file
    AO_filename_maker(1);
    std::string putf = AO_names_[1];
    
    // Initialize file (need to create it first for r+b access)
    {
        FILE* fp = fopen(std::get<0>(files_[putf]).c_str(), "wb");
        // Write zeros to initialize file to correct size
        std::vector<double> zeros(total_pq_sparse, 0.0);
        fwrite(zeros.data(), sizeof(double), total_pq_sparse, fp);
        fclose(fp);
    }
    
    // Process Q-blocks
    for (size_t Qstart = 0; Qstart < naux_; Qstart += Qblock) {
        size_t Qend = std::min(Qstart + Qblock, naux_);
        size_t Qsize = Qend - Qstart;
        
        // Read Q-major block from JK file
        psio_address addr = psio_get_address(PSIO_ZERO,
            Qstart * jk_n_function_pairs_ * sizeof(double));
        psio->read(jk_unit_, "(Q|mn) Integrals",
                  (char*)Qp, sizeof(double) * Qsize * jk_n_function_pairs_,
                  addr, &addr);
        
        // Transpose to p-major, processing each p
        for (size_t p = 0; p < nbf_; p++) {
            size_t num_q = small_skips_[p];
            if (num_q == 0) continue;
            
            // Clear p buffer
            std::fill(pp, pp + Qsize * num_q, 0.0);
            
            // Map each of DFHelper's sparse q indices to JK's packed index
            size_t q_sparse = 0;
            for (size_t q = 0; q < nbf_; q++) {
                if (!schwarz_fun_index_[p * nbf_ + q]) continue;
                
                // Find (p,q) in JK's triangular packing
                size_t m = std::max(p, q);
                size_t n = std::min(p, q);
                size_t tri_idx = m * (m + 1) / 2 + n;
                
                long int jk_idx = -1;
                if (tri_idx < jk_function_pairs_to_dense_.size()) {
                    jk_idx = jk_function_pairs_to_dense_[tri_idx];
                }
                
                if (jk_idx >= 0) {
                    // Copy this (p,q) data for all Q in block
                    for (size_t Q = 0; Q < Qsize; Q++) {
                        // Source: Qp[Q * jk_n_function_pairs_ + jk_idx]
                        // Dest:   pp[Q * num_q + q_sparse]
                        pp[Q * num_q + q_sparse] = Qp[Q * jk_n_function_pairs_ + jk_idx];
                    }
                }
                // else: leave as zero (screened by JK but not DFHelper)
                
                q_sparse++;
            }
            
            // Write this p's data to p-major file
            // File position: big_skips_[p] + Qstart * num_q
            size_t file_offset = big_skips_[p] + Qstart * num_q;
            put_tensor_AO(putf, pp, Qsize * num_q, file_offset, "r+b");
        }
        
        if (print_lvl_ > 1) {
            outfile->Printf("      Processed Q = %zu to %zu\n", Qstart, Qend - 1);
        }
    }
    
    psio->close(jk_unit_, 1);  // Keep JK's file
    
    // Mark that AO integrals are now ready
    built_ = true;
    AO_core_ = false;  // Integrals are on disk, not in core
    
    if (print_lvl_ > 0) {
        outfile->Printf("    DFHelper: Transpose complete\n\n");
    }
    
    timer_off("DFH: transpose JK integrals");
}
```

#### Subgoal 3.2: Add OpenMP Parallelization

Optimize the inner loop for multi-threaded execution:

```cpp
// In transpose_jk_integrals(), replace the p-loop with:

// Allocate per-thread buffers
std::vector<std::unique_ptr<double[]>> pbuf_threads(nthreads_);
for (size_t t = 0; t < nthreads_; t++) {
    pbuf_threads[t] = std::make_unique<double[]>(Qsize * max_small_skip);
}

// Process p in parallel (but write sequentially to avoid file contention)
// Strategy: compute in parallel, collect results, write sequentially

std::vector<std::vector<double>> p_results(nbf_);

#pragma omp parallel for schedule(dynamic) num_threads(nthreads_)
for (size_t p = 0; p < nbf_; p++) {
    int thread = 0;
#ifdef _OPENMP
    thread = omp_get_thread_num();
#endif
    double* pp = pbuf_threads[thread].get();
    size_t num_q = small_skips_[p];
    if (num_q == 0) continue;
    
    // ... same mapping logic as before ...
    
    // Store result for sequential write
    p_results[p].assign(pp, pp + Qsize * num_q);
}

// Sequential write phase
for (size_t p = 0; p < nbf_; p++) {
    if (p_results[p].empty()) continue;
    size_t file_offset = big_skips_[p] + Qstart * small_skips_[p];
    put_tensor_AO(putf, p_results[p].data(), p_results[p].size(), file_offset, "r+b");
}
```

#### Subgoal 3.3: Add Progress Reporting for Large Jobs

```cpp
// In the Q-block loop:
if (print_lvl_ > 0 && naux_ > 1000) {
    size_t percent = (Qend * 100) / naux_;
    outfile->Printf("      Transpose progress: %zu%%\r", percent);
    outfile->Flush();
}
```

**Deliverables for Phase 3:**
- [ ] Core transpose implementation
- [ ] OpenMP parallelization
- [ ] Progress reporting
- [ ] Memory usage optimization
- [ ] Unit tests for transpose correctness

---

### Phase 4: Integration with DFHelper Workflow

**Goal**: Seamlessly integrate JK integrals into DFHelper's existing workflow.

#### Subgoal 4.1: Modify initialize() to Check for JK Integrals

```cpp
void DFHelper::initialize() {
    // ... existing initialization code ...
    
    // If JK screening was imported, we can skip integral computation
    if (jk_screening_imported_ && method_ == "STORE") {
        if (print_lvl_ > 0) {
            outfile->Printf("    Using imported JK integrals (skipping AO computation)\n");
        }
        // Transpose will be called explicitly or automatically
        return;
    }
    
    // ... rest of existing initialize() ...
}
```

#### Subgoal 4.2: Add Auto-Transpose Option

```cpp
// New method for convenience
void DFHelper::initialize_from_jk(std::shared_ptr<JK> jk) {
    // Import screening
    import_jk_screening(jk);
    
    // Set method to STORE (use disk-based integrals)
    set_method("STORE");
    
    // Initialize (will skip integral computation)
    initialize();
    
    // Transpose JK integrals to p-major format
    transpose_jk_integrals();
}
```

#### Subgoal 4.3: Ensure transform() Works with Imported Integrals

The existing `transform()` method should work unchanged since it uses `grab_AO()` which reads from the p-major file we created. Verify this with testing.

#### Subgoal 4.4: Add Cleanup Method

```cpp
void DFHelper::clear_jk_import() {
    jk_screening_imported_ = false;
    jk_function_pairs_.clear();
    jk_function_pairs_to_dense_.clear();
    jk_n_function_pairs_ = 0;
    jk_unit_ = 0;
}
```

**Deliverables for Phase 4:**
- [ ] Modified initialize() with JK awareness
- [ ] Convenience method initialize_from_jk()
- [ ] Verification that transform() works correctly
- [ ] Cleanup method for JK import data
- [ ] Integration tests

---

### Phase 5: SAPT Integration

**Goal**: Modify FISAPT and FDDS dispersion code to use the new JK integral reuse capability.

#### Subgoal 5.1: Target Files for Integration

Key files (focused scope):
- `fisapt/fisapt.cc` - Uses JK object (received via `coulomb()` method at line 710) and multiple DFHelper instances
- `libsapt_solver/fdds_disp.cc` - Uses DFHelper at line 160 (for SAPT-DFT dispersion)

**Note**: We are NOT targeting `usapt0.cc` or `exch-ind30.cc` in this phase - those use different workflows.

#### Subgoal 5.2: Understanding FISAPT's JK/DFHelper Pattern

FISAPT has a unique pattern worth understanding:

```cpp
// fisapt.cc line 710-715: JK is passed in from external caller
void FISAPT::coulomb(std::shared_ptr<JK> jk) {
    // Reuse the passed JK object
    jk_ = jk;
    // ... uses jk_ throughout the class
}

// fisapt.cc line 4376-4377: DFHelper created separately for dispersion
auto dfh(std::make_shared<DFHelper>(primary_, auxiliary));
dfh->set_method("DIRECT_iaQ");
// ... transformations for dispersion terms
```

The JK object is created externally (often by the calling Python/C++ code) and passed to FISAPT.
DFHelper instances are created internally for integral transformations.

#### Subgoal 5.3: Modify FISAPT to Pass JK to DFHelper

Add a method to enable JK integral reuse in FISAPT:

```cpp
// In fisapt.h, add to FISAPT class:
class FISAPT {
protected:
    // ... existing members ...
    bool reuse_jk_integrals_ = false;  // New flag
    
public:
    /// Enable reuse of JK integrals for DFHelper transformations
    void set_reuse_jk_integrals(bool reuse) { reuse_jk_integrals_ = reuse; }
};
```

Modify DFHelper creation in `fisapt.cc` (e.g., around line 4376):

```cpp
// CURRENT code:
auto dfh(std::make_shared<DFHelper>(primary_, auxiliary));
dfh->set_memory(doubles_ - Cs[0]->nrow() * ncol);
dfh->set_method("DIRECT_iaQ");
dfh->set_nthreads(nT);
dfh->initialize();

// NEW code:
auto dfh(std::make_shared<DFHelper>(primary_, auxiliary));
dfh->set_memory(doubles_ - Cs[0]->nrow() * ncol);
dfh->set_nthreads(nT);

// Try to reuse JK integrals if enabled and compatible
if (reuse_jk_integrals_ && jk_ && can_reuse_jk_integrals(jk_, auxiliary)) {
    dfh->initialize_from_jk(jk_);
    outfile->Printf("    Reusing JK integrals for DFHelper transformations\n");
} else {
    dfh->set_method("DIRECT_iaQ");
    dfh->initialize();
}
```

#### Subgoal 5.4: Modify FDDS_Disp to Accept JK Object

FDDS_Disp (in `fdds_disp.cc`) currently creates its own DFHelper but doesn't have access to a JK object.
We need to add an optional JK parameter:

```cpp
// In fdds_disp.h, modify constructor or add setter:
class FDDS_Disp {
protected:
    std::shared_ptr<JK> jk_;  // Optional JK for integral reuse
    
public:
    /// Set JK object for integral reuse (optional)
    void set_jk(std::shared_ptr<JK> jk) { jk_ = jk; }
};
```

Modify DFHelper creation in `fdds_disp.cc` (around line 159):

```cpp
// CURRENT code (line 159-169):
dfh_ = std::make_shared<DFHelper>(primary_, auxiliary_);
dfh_->set_memory(doubles);
if (is_hybrid_) {
    dfh_->set_method("DIRECT");
} else {
    dfh_->set_method("DIRECT_iaQ");
}
dfh_->set_nthreads(nthread);
dfh_->set_metric_pow(0.0);
dfh_->initialize();

// NEW code:
dfh_ = std::make_shared<DFHelper>(primary_, auxiliary_);
dfh_->set_memory(doubles);
dfh_->set_nthreads(nthread);
dfh_->set_metric_pow(0.0);

// Try to reuse JK integrals if available
if (jk_ && can_reuse_jk_integrals(jk_, auxiliary_)) {
    dfh_->initialize_from_jk(jk_);
    outfile->Printf("    FDDS_Disp: Reusing JK integrals\n");
} else {
    if (is_hybrid_) {
        dfh_->set_method("DIRECT");
    } else {
        dfh_->set_method("DIRECT_iaQ");
    }
    dfh_->initialize();
}
```

#### Subgoal 5.5: Helper Function for Compatibility Check

Add a utility function (can be in dfhelper.h or a common header):

```cpp
/// Check if JK integrals can be reused for a given auxiliary basis
inline bool can_reuse_jk_integrals(std::shared_ptr<JK> jk, 
                                    std::shared_ptr<BasisSet> target_aux) {
    // Must be DiskDFJK
    auto disk_jk = std::dynamic_pointer_cast<DiskDFJK>(jk);
    if (!disk_jk) return false;
    
    // Integrals must be on disk
    if (!disk_jk->integrals_on_disk()) return false;
    
    // Auxiliary basis must match
    auto jk_aux = disk_jk->auxiliary();
    if (jk_aux->nbf() != target_aux->nbf()) return false;
    if (jk_aux->name() != target_aux->name()) return false;
    
    return true;
}
```

#### Subgoal 5.6: Python-Level Integration

Ensure the Python SAPT driver can enable this feature:

```python
# In psi4/driver/procrouting/proc.py or sapt.py
def run_fisapt(name, **kwargs):
    # ... existing setup ...
    
    # Build JK with SAVE option for integral reuse
    if core.get_option('FISAPT', 'REUSE_JK_INTEGRALS'):
        jk.set_df_ints_io('SAVE')
    
    # ... pass jk to FISAPT ...
```

**Deliverables for Phase 5:**
- [ ] Modified fisapt.cc with JK reuse capability
- [ ] Modified fdds_disp.cc with optional JK acceptance
- [ ] Helper function for basis compatibility checking
- [ ] FISAPT option REUSE_JK_INTEGRALS added
- [ ] Python driver modifications (if needed)
- [ ] FISAPT regression tests pass
- [ ] Performance benchmarks showing improvement

---

### Phase 6: Testing and Validation

**Goal**: Comprehensive testing to ensure correctness and measure performance gains.

#### Subgoal 6.1: Unit Tests for New Methods

```cpp
// test_dfhelper_jk_import.cc

TEST(DFHelperJKImport, BasicImport) {
    // Setup small test system
    auto mol = /* water molecule */;
    auto primary = /* cc-pVDZ */;
    auto auxiliary = /* cc-pVDZ-JKFIT */;
    
    // Build and run JK
    auto jk = std::make_shared<DiskDFJK>(primary, auxiliary, options);
    jk->set_df_ints_io("SAVE");
    jk->initialize();
    // ... compute J/K ...
    
    // Import into DFHelper
    auto dfh = std::make_shared<DFHelper>(primary, auxiliary);
    ASSERT_NO_THROW(dfh->import_jk_screening(jk));
    ASSERT_TRUE(dfh->has_jk_screening());
}

TEST(DFHelperJKImport, TransposeCorrectness) {
    // Compute integrals both ways, compare
    // ... JK method ...
    // ... DFHelper direct method ...
    // Compare transformed (ia|Q) tensors - should match
}

TEST(DFHelperJKImport, ScreeningMismatch) {
    // Test with different cutoffs - should warn but not fail
}
```

#### Subgoal 6.2: SAPT Regression Tests

Run existing SAPT test suite to ensure no numerical changes:
- `sapt1` through `sapt5` tests
- `fisapt` tests
- Compare energies to reference values

#### Subgoal 6.3: Performance Benchmarks

Create benchmark script:

```python
# benchmark_sapt_jk_reuse.py
import time
import psi4

molecules = {
    'water_dimer': """...""",
    'benzene_dimer': """...""",
    # etc
}

for name, mol in molecules.items():
    psi4.geometry(mol)
    
    # Without JK reuse
    psi4.set_options({'sapt_reuse_jk_integrals': False})
    t1 = time.time()
    psi4.energy('fisapt0')
    t_without = time.time() - t1
    
    # With JK reuse
    psi4.set_options({'sapt_reuse_jk_integrals': True})
    t2 = time.time()
    psi4.energy('fisapt0')
    t_with = time.time() - t2
    
    print(f"{name}: {t_without:.2f}s -> {t_with:.2f}s ({100*(t_without-t_with)/t_without:.1f}% faster)")
```

**Deliverables for Phase 6:**
- [ ] Unit tests for all new methods
- [ ] SAPT regression tests pass
- [ ] Performance benchmark results
- [ ] Documentation of expected speedups

---

### Phase 7: Documentation and Cleanup

**Goal**: Document the new functionality and ensure code quality.

#### Subgoal 7.1: Update DFHelper Doxygen Comments

Add comprehensive documentation for new methods in `dfhelper.h`.

#### Subgoal 7.2: Update SAPT Documentation

Document the new `SAPT_REUSE_JK_INTEGRALS` option and when it applies.

#### Subgoal 7.3: Add Developer Notes

Document the design decisions and limitations for future maintainers.

#### Subgoal 7.4: Code Review Preparation

- Ensure consistent code style
- Remove debug print statements
- Add appropriate timer calls
- Verify all edge cases handled

**Deliverables for Phase 7:**
- [ ] Doxygen comments complete
- [ ] User documentation updated
- [ ] Developer notes added
- [ ] Code review checklist complete

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Screening incompatibility | Medium | Medium | Validation with warnings |
| Numerical differences | Low | High | Comprehensive regression tests |
| Memory issues in transpose | Low | Medium | Careful blocking logic |
| PSIO file conflicts | Low | Medium | Use separate unit numbers |
| Performance regression | Low | High | Benchmark before/after |

---

## Timeline Estimate

| Phase | Estimated Effort | Dependencies |
|-------|------------------|--------------|
| Phase 1: DiskDFJK Accessors | 2-3 hours | None |
| Phase 2: DFHelper Import | 3-4 hours | Phase 1 |
| Phase 3: Transpose Implementation | 4-6 hours | Phase 2 |
| Phase 4: DFHelper Integration | 2-3 hours | Phase 3 |
| Phase 5: SAPT Integration | 3-4 hours | Phase 4 |
| Phase 6: Testing | 4-6 hours | Phase 5 |
| Phase 7: Documentation | 2-3 hours | Phase 6 |

**Total: ~20-29 hours of development time**

---

## Success Criteria

1. **Correctness**: SAPT energies unchanged (within numerical precision)
2. **Performance**: Measurable speedup for SAPT calculations (target: 20-40% for integral-bound jobs)
3. **Robustness**: Graceful handling of edge cases (different basis sets, screening mismatches)
4. **Maintainability**: Clean code with good documentation
5. **Compatibility**: No breaking changes to existing DFHelper or JK interfaces

---

## Future Enhancements (Out of Scope)

- Support for MemDFJK integral reuse (currently in-core only)
- Support for wK integral reuse (range-separated functionals)
- Automatic basis set matching heuristics
- Integration with other post-HF methods (MP2, CC)

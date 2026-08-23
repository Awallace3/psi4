"""Anisotropic `Cn[l1 k1, l2 k2, j]` parity against the partition-matched ISA-GRID oracle.

This is plan task B10. Every reference number below is a hard-coded literal, extracted once
from the approved reference tree's `work/H2O-isagrid/H2O_ref_wt4_L3_casimir.out`, sha256
`a405b6b07904beecdb7d7fb7a527cc64d0b92355212261a8c5b349a09c296999`, through
`devtools.camcasp_reference.parse_anisotropic_cn`, and reviewed. This module must never read
that data at runtime, deserialise it, or invoke any external command;
`test_module_has_no_runtime_reference_dependency` at the bottom enforces that.

The oracle is the **ISA-GRID** CASIMIR run -- the same partition family as this pipeline, and
the same calculation whose isotropic `00 00 0` row is already gated as `ISA_GRID_C6`..`C12` in
`test_atomic_polarizabilities.py`. Comparing against the reviewed **DF** run instead would be
comparing two different models; see that module's docstring.

WHAT THIS MODULE MEASURES, AND WHY IT IS XFAIL
==============================================

The comparison misses the plan's `rtol=1e-4` gate on every one of the 10457 shared nonzero
coefficients. It is kept, at that gate, as a strict xfail, exactly as the six `DF_*`
comparisons are: the value here is the localisation, not a green tick. Three separate
findings came out of the measurement and are recorded in the xfail reasons in numbers.

1. **It is not a sign convention.** The two tables really do use different exchange
   conventions -- ours satisfies `C[l2 k2, l1 k1, j] = (-1)^(l1+l2) C[l1 k1, l2 k2, j]`
   (0 violations, see `test_our_table_obeys_the_derivations_exchange_law`) and CASIMIR's
   homonuclear blocks are plain-symmetric (0 violations, see
   `test_the_oracle_c6_block_is_plain_symmetric`) -- and on `l1+l2` odd labels the two
   therefore disagree by a full sign. But *no* sign function repairs the table. Over the
   shared set the identity leaves 5343 sign disagreements out of 10457; `(-1)^l1` leaves
   5158, `(-1)^l2` 5134, `(-1)^(l1+l2)` 5295, `(-1)^j` 5263, and the `i^(l1-l2-j)`/`Ncal`
   ratio of the derivation's §4.4 leaves 5189. The best of them explains 51.1 percent of the
   observed flips. Decisively, the required sign is not a function of `(l1, l2, j)` at all:
   114 of the 128 populated triples carry *both* signs, and 238 of the 273 populated
   `(n, l1, l2, j)` groups do, so no S-function phase convention -- which by construction is
   one real sign per `(l1, l2, j)` -- can be the explanation. At a cut of one percent of the
   per-order scale, C7 still has 14 of its 16 triples mixed, so this is not cancellation
   noise. Note also that `(-1)^(l1+l2)` is *symmetric* under the label exchange and so cannot
   change a table's exchange symmetry at all; `(-1)^l1` and `(-1)^l2` can, and differ from
   each other exactly on the `l1+l2` odd labels.

2. **There is a real convention difference, and it is a magnitude one.** At C6 -- the only
   published order at which exactly one site-block quadruple `(la,la',lb,lb') = (1,1,1,1)`
   contributes, so no cancellation between blocks is possible -- there are 0 sign
   disagreements in all 75 entries, and the ratio ours/CASIMIR at fixed `(l1, l2) = (2, 2)`
   varies with `j` exactly as `1 / |<l1 0; l2 0 | j 0>|`. Multiplying ours by that
   Clebsch-Gordan coefficient makes the ratio `j`-independent to **6.5e-07** -- the printed
   precision of the reference -- for every one of the 4 + 6 + 9 component pairs `(k1, k2)` in
   the three blocks. `test_the_c6_j_dependence_is_the_measured_clebsch_gordan_factor` gates
   that, because it is a sharp statement about our recoupling table's `j` dependence and a
   regression in it must be loud.

   That factor **cannot be adopted as a convention**: `<l1 0; l2 0 | j 0>` vanishes
   identically on the 2968 of 10457 shared nonzero entries with `l1+l2+j` odd, where both
   tables print nonzero values (largest such: O-O C12 `22s 30 4`, CASIMIR -2194.16 against
   ours -276.144). So the residual is not merely a convention, and that is the headline
   result of B10. See §9.1 of `2026-08-18-anisotropic-recoupling-derivation.md`, which
   already flagged the residual real sign and normalisation per `(l1, l2, j)` as unverified.

3. **What is left after the C6 factor is the known property deficit, but only on H-H.**
   Corrected residuals at C6 are 0.0993 / 0.0156 / 0.0761 on H-H at `(l1,l2)` = (0,0) /
   (0,2) / (2,2) -- inside the recorded 1-10 percent ISA-GRID C6 band, and factorising into
   a per-component product to 2.2e-04 -- but 0.0562 / 0.0318 / 0.632 / 0.783 on H-O and
   0.0117 / 0.714 / 1.503 on O-O. The excess sits entirely on labels carrying a rank-2
   coupled index on the **O** site. That is cancellation, not a separate defect, and it is
   measurable in the dipole block that feeds C6: O's `alpha` components agree with the same
   oracle to 1.7 / 0.8 / 2.4 percent and its isotropic mean to 1.6 percent, yet its coupled
   rank-2 part `q20 = (2 a_zz - a_xx - a_yy) / 2` is -0.212260 against -0.128916, **64.7
   percent** out, because O is nearly isotropic and `q20` is a small difference of large
   numbers. H, which is strongly anisotropic, is 15.3 percent out on `a_yy` and only 0.85
   percent out on `q20`. The 1.714 ratio the O-O `(0,2)` C6 labels show is that 64.7 percent
   to within 4 percent. The isotropic `00 00 0` entries of this
   very table reproduce the recorded ISA-GRID bands exactly -- 0.0117 / 0.0562 / 0.0993 at
   C6 and 0.347 / 0.411 / 0.456 at C12 -- so the run and the extraction are sound and the
   anisotropic excess is a property of the anisotropic sector alone.

Frames, and the one way to get this silently wrong
==================================================

`ATOMIC DISPERSION COEFFICIENTS` is indexed by *ordered site pairs*, row `A * 3 + B`, with
sites in the reviewed order O, H1, H2 and **every site frame the molecular frame**. CASIMIR
prints one block per site *type* in each site's own local axes, and
`H2O.axes` gives `H1  z global Z x from H2 to H1` / `H2  z global Z x from H1 to H2`: with H1
at negative x, H2's local axes are the molecular axes and H1's are those rotated by pi about
z. So the comparable rows are `(O, O)`, `(H2, O)` and `(H2, H2)` and no rotation is needed.
Using H1 instead multiplies each coefficient by `(-1)^(|m1| + |m2|)` and silently flips the
sign of rather more than half the table. `test_h1_and_h2_rows_differ_by_the_pi_rotation`
verifies the rotation rather than assuming it, so the row choice above is evidenced.

Selection rule for the literals
===============================

Two rule-defined subsets, both fixed by the *reference* before any comparison was made, so
neither can be a cherry-pick:

* `ISA_GRID_ANISOTROPIC_C6` -- **every** label CASIMIR prints with a nonzero C6, all 17 + 24
  + 34 = 75 of them, i.e. the complete C6 sub-table. C6 is the order that isolates the
  single `(1,1,1,1)` site block, which is what makes finding 2 above possible.
* `ISA_GRID_ANISOTROPIC_LEADING` -- for each block and each order C7..C12, the single label
  carrying the **largest** `|Cn|` in the reference. Eighteen entries, and by construction the
  ones that dominate the dispersion energy at their order, which is the opposite of selecting
  for a flattering result.

Embedding all 10457 shared coefficients was rejected; the whole-table statistics are recorded
in the xfail reasons and in docs/superpowers/specs/2026-08-18-isa-grid-oracle.md.
"""

import inspect
import math
import os
from pathlib import Path

import numpy as np
import pytest

import psi4
from psi4.driver.procrouting import atomic_polarizability as native_driver

from test_atomic_polarizabilities import (
    PARITY_PROTOCOL,
    PARITY_SKIP_REASON,
    REVIEWED_GEOMETRY,
    TENSOR_ATOL,
    TENSOR_RTOL,
)

pytestmark = [pytest.mark.psi, pytest.mark.api, pytest.mark.mints]

#: `ATOMIC DISPERSION LABELS` columns: n, l1, k1, l2, k2, j. `k` indexes the per-rank real
#: component order `l0, l1c, l1s, l2c, l2s, ...`, the same convention CASIMIR's `22c`/`32s`
#: label fields carry.
ORDERS = (6, 7, 8, 9, 10, 11, 12)

#: Which ordered site-pair row of our table each CASIMIR type-pair block compares against.
#: See the frame discussion in the module docstring: H2, not H1.
ORACLE_ROW = {("O", "O"): 0, ("H", "O"): 6, ("H", "H"): 8}

#: `<l1 0; l2 0 | j 0>` for l1 = l2 = 2, the only triple the C6 sub-table pins (finding 2).
CG_2_2 = {0: 1.0 / math.sqrt(5.0), 2: -math.sqrt(2.0 / 7.0), 4: math.sqrt(18.0 / 35.0)}

# --------------------------------------------------------------------------------------
# Reviewed literals, ISA-GRID CASIMIR anisotropic table. Keys are (l1, k1, l2, k2, j);
# the trailing comment gives CASIMIR's own label spelling, the value measured at
# PARITY_PROTOCOL on 2026-08-20, and the relative deviation.
# --------------------------------------------------------------------------------------

ISA_GRID_ANISOTROPIC_C6 = {
    # CASIMIR block "O O" -- 17 labels with a nonzero C6.
    ('O', 'O'): {
        (0, 0, 0, 0, 0): 26.48177,              # 00 00 0, ours 26.1715, rel 0.012
        (0, 0, 2, 0, 2): -0.124485,             # 00 20 2, ours -0.190842, rel 0.533
        (0, 0, 2, 3, 2): -0.09122333,           # 00 22c 2, ours -0.172838, rel 0.895
        (2, 0, 0, 0, 2): -0.124485,             # 20 00 2, ours -0.190842, rel 0.533
        (2, 0, 2, 0, 0): 0.0001211599,          # 20 20 0, ours 0.000673942, rel 4.562
        (2, 0, 2, 0, 2): 0.0001730855,          # 20 20 2, ours 0.000805514, rel 3.654
        (2, 0, 2, 0, 4): 0.001869324,           # 20 20 4, ours 0.00648427, rel 2.469
        (2, 0, 2, 3, 0): 9.442612e-05,          # 20 22c 0, ours 0.000649701, rel 5.881
        (2, 0, 2, 3, 2): 0.0001348945,          # 20 22c 2, ours 0.000776542, rel 4.757
        (2, 0, 2, 3, 4): 0.00145686,            # 20 22c 4, ours 0.00625104, rel 3.291
        (2, 3, 0, 0, 2): -0.09122333,           # 22c 00 2, ours -0.172838, rel 0.895
        (2, 3, 2, 0, 0): 9.442612e-05,          # 22c 20 0, ours 0.000649701, rel 5.881
        (2, 3, 2, 0, 2): 0.0001348945,          # 22c 20 2, ours 0.000776542, rel 4.757
        (2, 3, 2, 0, 4): 0.00145686,            # 22c 20 4, ours 0.00625104, rel 3.291
        (2, 3, 2, 3, 0): 0.0003960647,          # 22c 22c 0, ours 0.00121389, rel 2.065
        (2, 3, 2, 3, 2): 0.0005658067,          # 22c 22c 2, ours 0.00145088, rel 1.564
        (2, 3, 2, 3, 4): 0.006110712,           # 22c 22c 4, ours 0.0116793, rel 0.911
    },
    # CASIMIR block "H O" -- 24 labels with a nonzero C6.
    ('H', 'O'): {
        (0, 0, 0, 0, 0): 4.142317,              # 00 00 0, ours 3.90955, rel 0.056
        (0, 0, 2, 0, 2): -0.01971193,           # 00 20 2, ours -0.0290104, rel 0.472
        (0, 0, 2, 3, 2): -0.01452666,           # 00 22c 2, ours -0.0260489, rel 0.793
        (2, 0, 0, 0, 2): 0.07047388,            # 20 00 2, ours 0.0723593, rel 0.027
        (2, 0, 2, 0, 0): -6.693689e-05,         # 20 20 0, ours -0.000239094, rel 2.572
        (2, 0, 2, 0, 2): -9.562413e-05,         # 20 20 2, ours -0.000285771, rel 1.988
        (2, 0, 2, 0, 4): -0.001032741,          # 20 20 4, ours -0.00230042, rel 1.227
        (2, 0, 2, 3, 0): -5.885488e-05,         # 20 22c 0, ours -0.000234279, rel 2.981
        (2, 0, 2, 3, 2): -8.407841e-05,         # 20 22c 2, ours -0.000280017, rel 2.330
        (2, 0, 2, 3, 4): -0.0009080468,         # 20 22c 4, ours -0.00225409, rel 1.482
        (2, 1, 0, 0, 2): -1.209746,             # 21c 00 2, ours -1.23002, rel 0.017
        (2, 1, 2, 0, 0): 0.001165109,           # 21c 20 0, ours 0.0041683, rel 2.578
        (2, 1, 2, 0, 2): 0.001664441,           # 21c 20 2, ours 0.00498207, rel 1.993
        (2, 1, 2, 0, 4): 0.01797596,            # 21c 20 4, ours 0.0401049, rel 1.231
        (2, 1, 2, 3, 0): 0.0008846701,          # 21c 22c 0, ours 0.00378112, rel 3.274
        (2, 1, 2, 3, 2): 0.001263814,           # 21c 22c 2, ours 0.0045193, rel 2.576
        (2, 1, 2, 3, 4): 0.0136492,             # 21c 22c 4, ours 0.0363797, rel 1.665
        (2, 3, 0, 0, 2): 0.8535328,             # 22c 00 2, ours 0.897757, rel 0.052
        (2, 3, 2, 0, 0): -0.0008197977,         # 22c 20 0, ours -0.00303479, rel 2.702
        (2, 3, 2, 0, 2): -0.00117114,           # 22c 20 2, ours -0.00362727, rel 2.097
        (2, 3, 2, 0, 4): -0.01264831,           # 22c 20 4, ours -0.029199, rel 1.309
        (2, 3, 2, 3, 0): -0.0005476239,         # 22c 22c 0, ours -0.0026381, rel 3.817
        (2, 3, 2, 3, 2): -0.0007823198,         # 22c 22c 2, ours -0.00315313, rel 3.030
        (2, 3, 2, 3, 4): -0.008449054,          # 22c 22c 4, ours -0.0253822, rel 2.004
    },
    # CASIMIR block "H H" -- 34 labels with a nonzero C6.
    ('H', 'H'): {
        (0, 0, 0, 0, 0): 0.6514697,             # 00 00 0, ours 0.586777, rel 0.099
        (0, 0, 2, 0, 2): 0.01107489,            # 00 20 2, ours 0.0108331, rel 0.022
        (0, 0, 2, 1, 2): -0.191219,             # 00 21c 2, ours -0.18554, rel 0.030
        (0, 0, 2, 3, 2): 0.1348116,             # 00 22c 2, ours 0.135444, rel 0.005
        (2, 0, 0, 0, 2): 0.01107489,            # 20 00 2, ours 0.0108331, rel 0.022
        (2, 0, 2, 0, 0): 3.797152e-05,          # 20 20 0, ours 9.02463e-05, rel 1.377
        (2, 0, 2, 0, 2): 5.424503e-05,          # 20 20 2, ours 0.000107865, rel 0.988
        (2, 0, 2, 0, 4): 0.0005858463,          # 20 20 4, ours 0.000868297, rel 0.482
        (2, 0, 2, 1, 0): -0.0006502418,         # 20 21c 0, ours -0.00152965, rel 1.352
        (2, 0, 2, 1, 2): -0.0009289169,         # 20 21c 2, ours -0.00182828, rel 0.968
        (2, 0, 2, 1, 4): -0.0100323,            # 20 21c 4, ours -0.0147173, rel 0.467
        (2, 0, 2, 3, 0): 0.0004562339,          # 20 22c 0, ours 0.00111266, rel 1.439
        (2, 0, 2, 3, 2): 0.0006517628,          # 20 22c 2, ours 0.00132989, rel 1.040
        (2, 0, 2, 3, 4): 0.007039038,           # 20 22c 4, ours 0.0107054, rel 0.521
        (2, 1, 0, 0, 2): -0.191219,             # 21c 00 2, ours -0.18554, rel 0.030
        (2, 1, 2, 0, 0): -0.0006502418,         # 21c 20 0, ours -0.00152965, rel 1.352
        (2, 1, 2, 0, 2): -0.0009289169,         # 21c 20 2, ours -0.00182828, rel 0.968
        (2, 1, 2, 0, 4): -0.0100323,            # 21c 20 4, ours -0.0147173, rel 0.467
        (2, 1, 2, 1, 0): 0.01127953,            # 21c 21c 0, ours 0.0263889, rel 1.340
        (2, 1, 2, 1, 2): 0.01611361,            # 21c 21c 2, ours 0.0315407, rel 0.957
        (2, 1, 2, 1, 4): 0.174027,              # 21c 21c 4, ours 0.253898, rel 0.459
        (2, 1, 2, 3, 0): -0.007942113,          # 21c 22c 0, ours -0.0192549, rel 1.424
        (2, 1, 2, 3, 2): -0.01134588,           # 21c 22c 2, ours -0.023014, rel 1.028
        (2, 1, 2, 3, 4): -0.1225355,            # 21c 22c 4, ours -0.185259, rel 0.512
        (2, 3, 0, 0, 2): 0.1348116,             # 22c 00 2, ours 0.135444, rel 0.005
        (2, 3, 2, 0, 0): 0.0004562339,          # 22c 20 0, ours 0.00111266, rel 1.439
        (2, 3, 2, 0, 2): 0.0006517628,          # 22c 20 2, ours 0.00132989, rel 1.040
        (2, 3, 2, 0, 4): 0.007039038,           # 22c 20 4, ours 0.0107054, rel 0.521
        (2, 3, 2, 1, 0): -0.007942113,          # 22c 21c 0, ours -0.0192549, rel 1.424
        (2, 3, 2, 1, 2): -0.01134588,           # 22c 21c 2, ours -0.023014, rel 1.028
        (2, 3, 2, 1, 4): -0.1225355,            # 22c 21c 4, ours -0.185259, rel 0.512
        (2, 3, 2, 3, 0): 0.005610271,           # 22c 22c 0, ours 0.0140729, rel 1.508
        (2, 3, 2, 3, 2): 0.008014673,           # 22c 22c 2, ours 0.0168203, rel 1.099
        (2, 3, 2, 3, 4): 0.08655847,            # 22c 22c 4, ours 0.135401, rel 0.564
    },
}

ISA_GRID_ANISOTROPIC_LEADING = {
    ('O', 'O'): {
        (7, 0, 0, 1, 0, 1): -21.90433,          # 00 10 1, ours 11.959, rel 1.546
        (8, 0, 0, 0, 0, 0): 490.4584,           # 00 00 0, ours 393.484, rel 0.198
        (9, 0, 0, 1, 0, 1): -376.2563,          # 00 10 1, ours 178.901, rel 1.475
        (10, 0, 0, 0, 0, 0): 9673.248,          # 00 00 0, ours 7129.89, rel 0.263
        (11, 0, 0, 1, 0, 1): -6171.903,         # 00 10 1, ours 2909.61, rel 1.471
        (12, 0, 0, 0, 0, 0): 150417.4,          # 00 00 0, ours 98233.5, rel 0.347
    },
    ('H', 'O'): {
        (7, 1, 0, 0, 0, 1): 4.149667,           # 10 00 1, ours -2.01379, rel 1.485
        (8, 0, 0, 0, 0, 0): 65.08315,           # 00 00 0, ours 50.1634, rel 0.229
        (9, 1, 1, 0, 0, 1): -133.5643,          # 11c 00 1, ours -7.64913, rel 0.943
        (10, 0, 0, 0, 0, 0): 1262.305,          # 00 00 0, ours 870.869, rel 0.310
        (11, 1, 1, 0, 0, 1): -2559.7,           # 11c 00 1, ours -174.567, rel 0.932
        (12, 0, 0, 0, 0, 0): 18759.28,          # 00 00 0, ours 11047.9, rel 0.411
    },
    ('H', 'H'): {
        (7, 0, 0, 1, 0, 1): 0.6712108,          # 00 10 1, ours 0.3101, rel 0.538
        (8, 0, 0, 0, 0, 0): 8.463255,           # 00 00 0, ours 6.30312, rel 0.255
        (9, 0, 0, 1, 1, 1): -19.16728,          # 00 11c 1, ours 1.02839, rel 1.054
        (10, 0, 0, 0, 0, 0): 168.1889,          # 00 00 0, ours 107.755, rel 0.359
        (11, 0, 0, 1, 1, 1): -299.0681,         # 00 11c 1, ours 15.166, rel 1.051
        (12, 0, 0, 0, 0, 0): 2278.796,          # 00 00 0, ours 1240.71, rel 0.456
    },
}


# --------------------------------------------------------------------------------------
# The run.
# --------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def parity_anisotropic():
    """`(labels, coefficients)` from the reviewed parity protocol under PARTITION=ISA."""
    if os.environ.get("PSI4_ATOMIC_POLARIZABILITY_PARITY") != "1":
        pytest.skip(PARITY_SKIP_REASON)
    psi4.core.clean_variables()
    psi4.core.be_quiet()
    psi4.set_options({"atomic_polarizability_partition": "ISA", **PARITY_PROTOCOL})
    wfn = native_driver.atomic_polarizabilities(molecule=psi4.geometry(REVIEWED_GEOMETRY))
    labels = np.asarray(wfn.array_variable("ATOMIC DISPERSION LABELS")).astype(int)
    coefficients = np.asarray(wfn.array_variable("ATOMIC DISPERSION COEFFICIENTS"))
    return labels, coefficients


def _column_index(labels):
    return {tuple(row): column for column, row in enumerate(labels.tolist())}


def _order_scale(labels, coefficients, row):
    """Largest `|Cn|` of each order in one ordered-site-pair row.

    Anisotropic coefficients of one order span many orders of magnitude and the small ones
    are differences of large ones, so an absolute floor has to be referred to the order, not
    to the entry. Used only to separate genuine values from cancellation noise, never to
    relax a comparison.
    """
    return {n: float(np.abs(coefficients[row][labels[:, 0] == n]).max()) for n in ORDERS}


# --------------------------------------------------------------------------------------
# Literal-only checks. These need no run and so are not gated on the parity variable.
# --------------------------------------------------------------------------------------


def test_the_oracle_c6_block_is_plain_symmetric():
    """CASIMIR's homonuclear blocks are symmetric under `(l1 k1) <-> (l2 k2)`, bit-for-bit.

    Half of the convention question is this fact, so it is asserted from the literals rather
    than asserted in prose. Our table obeys `(-1)^(l1+l2)` instead (next test), which is why
    they disagree by a full sign wherever `l1 + l2` is odd.
    """
    for pair in (("O", "O"), ("H", "H")):
        block = ISA_GRID_ANISOTROPIC_C6[pair]
        for (l1, k1, l2, k2, j), value in block.items():
            mirror = (l2, k2, l1, k1, j)
            assert mirror in block, (pair, mirror)
            assert block[mirror] == value, (pair, (l1, k1, l2, k2, j))


def test_the_literal_subsets_follow_their_stated_selection_rule():
    """The rules in the module docstring, made executable.

    C6 set: every label of the reference's C6 sub-table, which the rank algebra forces to
    have `l1, l2` in {0, 2} -- `la = la' = 1` is the only quadruple summing to `n - 2 = 4`,
    and an odd coupled rank needs `la != la'`. Leading set: one label per block per order
    C7..C12.
    """
    assert tuple(sorted(ISA_GRID_ANISOTROPIC_C6)) == (("H", "H"), ("H", "O"), ("O", "O"))
    assert [len(v) for _, v in sorted(ISA_GRID_ANISOTROPIC_C6.items())] == [34, 24, 17]
    for pair, block in ISA_GRID_ANISOTROPIC_C6.items():
        for (l1, k1, l2, k2, j), value in block.items():
            assert l1 in (0, 2) and l2 in (0, 2), (pair, (l1, k1, l2, k2, j))
            assert abs(l1 - l2) <= j <= l1 + l2
            assert j % 2 == 0                      # n = 6 is even and n === j (mod 2)
            assert value != 0.0
    for pair, block in ISA_GRID_ANISOTROPIC_LEADING.items():
        assert sorted(n for n, *_ in block) == [7, 8, 9, 10, 11, 12], pair
        for (n, l1, k1, l2, k2, j), value in block.items():
            assert (n - j) % 2 == 0, (pair, (n, l1, k1, l2, k2, j))
            assert value != 0.0


# --------------------------------------------------------------------------------------
# Conventions of our own table, verified rather than assumed.
# --------------------------------------------------------------------------------------

#: Fraction of the per-order scale below which a coefficient is cancellation noise. Measured:
#: with no floor our exchange law shows 68 / 568 apparent violations out of 16985 entries on
#: the (O,O) / (H2,H2) rows; at 1e-6 of the order scale it is 0 / 0 out of 1706 / 5921. The
#: floor is a noise floor, not a tolerance -- the surviving comparisons are made at 1e-8.
NOISE_FRACTION = 1.0e-6


@pytest.mark.parametrize("row", [0, 4, 8], ids=["O,O", "H1,H1", "H2,H2"])
def test_our_table_obeys_the_derivations_exchange_law(parity_anisotropic, row):
    """`C[l2 k2, l1 k1, j] = (-1)^(l1+l2) C[l1 k1, l2 k2, j]` on a diagonal site pair.

    §B.5 check 4 of the plan, and the reason our published array is over *ordered* site
    pairs. Recorded here because it is the half of the convention mismatch that belongs to
    us: CASIMIR's blocks are plain-symmetric.
    """
    labels, coefficients = parity_anisotropic
    column = _column_index(labels)
    scale = _order_scale(labels, coefficients, row)
    compared = 0
    for (n, l1, k1, l2, k2, j), value in zip(labels.tolist(), coefficients[row]):
        mirror = column.get((n, l2, k2, l1, k1, j))
        if mirror is None:
            continue
        other = float(coefficients[row][mirror])
        if max(abs(value), abs(other)) < NOISE_FRACTION * scale[n]:
            continue
        compared += 1
        expected = (-1.0) ** (l1 + l2) * value
        assert abs(other - expected) <= 1.0e-8 * max(abs(value), abs(other)), (
            n, l1, k1, l2, k2, j, value, other)
    assert compared > 1000


def test_h1_and_h2_rows_differ_by_the_pi_rotation(parity_anisotropic):
    """H1's local axes are the molecular axes turned by pi about z, so its rows differ from
    H2's by `(-1)^(|m1| + |m2|)`.

    This is what licenses comparing the `(H2, ...)` rows against CASIMIR's `H` blocks
    without rotating anything. Measured: 0 violations out of 5921 / 3195 / 3195 entries
    above the noise floor, worst relative deviation 1.65e-11.
    """
    labels, coefficients = parity_anisotropic

    def magnitude(k):
        return 0 if k == 0 else (k + 1) // 2

    cases = {"(H1,H1)/(H2,H2)": (4, 8, True, True),
             "(H1,O)/(H2,O)": (3, 6, True, False),
             "(O,H1)/(O,H2)": (1, 2, False, True)}
    for name, (left, right, first, second) in cases.items():
        scale = _order_scale(labels, coefficients, left)
        compared = 0
        for (n, l1, k1, l2, k2, j), a, b in zip(
                labels.tolist(), coefficients[left], coefficients[right]):
            if max(abs(a), abs(b)) < NOISE_FRACTION * scale[n]:
                continue
            compared += 1
            power = (magnitude(k1) if first else 0) + (magnitude(k2) if second else 0)
            assert abs(a - (-1.0) ** power * b) <= 1.0e-8 * max(abs(a), abs(b)), (
                name, n, l1, k1, l2, k2, j, a, b)
        assert compared > 500, name


CG_TOLERANCE = 1.0e-5


def test_the_c6_j_dependence_is_the_measured_clebsch_gordan_factor(parity_anisotropic):
    """The convention resolution of finding 2, gated.

    At C6 one site-block quadruple contributes, so at fixed `(l1, l2, k1, k2)` the ratio
    ours/CASIMIR over `j` measures the recoupling prefactor alone. Measured: multiplying ours
    by `<l1 0; l2 0 | j 0>` makes that ratio `j`-independent to 6.5e-07 for all 4 + 6 + 9
    component pairs of the three blocks. That is a sharp statement about our table's `j`
    dependence, so a regression in it must be loud -- but see the module docstring: the
    factor is *not* adoptable as a global convention, because it vanishes on 2968 of the
    10457 shared entries where both tables are nonzero.
    """
    labels, coefficients = parity_anisotropic
    column = _column_index(labels)
    checked = 0
    for pair, row in ORACLE_ROW.items():
        block = ISA_GRID_ANISOTROPIC_C6[pair]
        corrected = {}
        for (l1, k1, l2, k2, j), reference in block.items():
            if (l1, l2) != (2, 2):
                continue
            ours = float(coefficients[row][column[(6, l1, k1, l2, k2, j)]])
            corrected.setdefault((k1, k2), {})[j] = ours / reference * abs(CG_2_2[j])
        assert corrected, pair
        for (k1, k2), by_j in corrected.items():
            assert sorted(by_j) == [0, 2, 4], (pair, k1, k2)
            values = [by_j[j] for j in (0, 2, 4)]
            spread = (max(values) - min(values)) / abs(sum(values) / len(values))
            assert spread <= CG_TOLERANCE, (pair, k1, k2, values, spread)
            checked += 1
    assert checked == 4 + 6 + 9


# --------------------------------------------------------------------------------------
# The B10 comparison itself.
# --------------------------------------------------------------------------------------

C6_XFAIL_REASON = (
    "measured at PARITY_PROTOCOL under PARTITION=ISA on 2026-08-20 against the "
    "partition-matched ISA-GRID CASIMIR table: 0 of the 75 labels with a nonzero C6 is "
    "inside rtol=1e-4. Median relative deviation 1.23, worst 5.88 (O-O 22c 20 0, ours "
    "6.49701e-04 against 9.442612e-05); per block median/worst 2.47/5.88 O-O, 1.99/3.82 "
    "H-O, 0.963/1.51 H-H, against 0.0117/0.0562/0.0993 on the isotropic 00 00 0 entry of "
    "the same table, which is exactly the already-recorded ISA-GRID C6 band. The residual "
    "is not a sign convention -- 0 of the 75 has a sign disagreement, and over the full "
    "10457-entry shared set no sign function explains more than 51.1 percent of the flips "
    "while 114 of 128 (l1,l2,j) triples require both signs. It is a magnitude convention "
    "plus a property deficit: multiplying ours by |<l1 0; l2 0|j 0>| makes the ratio "
    "j-independent to 6.5e-07 across all 19 component pairs, after which the residual is "
    "0.0993/0.0156/0.0761 on H-H -- inside the recorded 1-10 percent band and factorising "
    "per component to 2.2e-04 -- but 0.632/0.783 on H-O and 0.714/1.503 on O-O, entirely "
    "on labels carrying a rank-2 coupled index on the O site, where a 1.7-2.4 percent "
    "per-component agreement in the nearly isotropic O dipole block amplifies into a 64.7 "
    "percent error in its coupled rank-2 part (q20 -0.212260 against -0.128916). That "
    "factor cannot be adopted, because it vanishes on the 2968 of 10457 shared entries with "
    "l1+l2+j odd where both tables are nonzero. Neither the gate nor a literal was altered. "
    "See docs/superpowers/specs/2026-08-18-isa-grid-oracle.md"
)

LEADING_XFAIL_REASON = (
    "measured at PARITY_PROTOCOL under PARTITION=ISA on 2026-08-20: 0 of the 18 "
    "leading-magnitude C7..C12 coefficients is inside rtol=1e-4; median relative deviation "
    "0.50, worst 1.55 (O-O C7 00 10 1, ours 11.959 against -21.90433). Six of the 18 have "
    "the opposite sign and every one of them is an odd-order label with l1 + l2 odd and "
    "j = 1, where our engine's (-1)^(l1+l2) label-exchange law and CASIMIR's plain label "
    "symmetry are in direct contradiction. The nine even-order entries are the isotropic "
    "00 00 0 coefficient and reproduce the recorded ISA-GRID bands exactly: 0.198/0.263/"
    "0.347 O-O, 0.229/0.310/0.411 H-O, 0.255/0.359/0.456 H-H at C8/C10/C12. So the even "
    "orders are the known rank deficit and the odd orders are dominated by the unresolved "
    "convention of the module docstring. Neither the gate nor a literal was altered. "
    "See docs/superpowers/specs/2026-08-18-isa-grid-oracle.md"
)


@pytest.mark.scf
@pytest.mark.xfail(strict=True, reason=C6_XFAIL_REASON)
def test_parity_anisotropic_c6_matches_the_isa_grid_oracle(parity_anisotropic):
    labels, coefficients = parity_anisotropic
    column = _column_index(labels)
    for pair, row in ORACLE_ROW.items():
        for label, reference in ISA_GRID_ANISOTROPIC_C6[pair].items():
            ours = float(coefficients[row][column[(6,) + label]])
            np.testing.assert_allclose(
                ours, reference, rtol=TENSOR_RTOL, atol=TENSOR_ATOL,
                err_msg=f"{pair} C6 label {label}")


@pytest.mark.scf
@pytest.mark.xfail(strict=True, reason=LEADING_XFAIL_REASON)
def test_parity_anisotropic_leading_coefficients_match_the_isa_grid_oracle(parity_anisotropic):
    labels, coefficients = parity_anisotropic
    column = _column_index(labels)
    for pair, row in ORACLE_ROW.items():
        for label, reference in ISA_GRID_ANISOTROPIC_LEADING[pair].items():
            ours = float(coefficients[row][column[label]])
            np.testing.assert_allclose(
                ours, reference, rtol=TENSOR_RTOL, atol=TENSOR_ATOL,
                err_msg=f"{pair} C{label[0]} label {label[1:]}")


# --------------------------------------------------------------------------------------
# The part of our table that has no oracle at all.
# --------------------------------------------------------------------------------------

#: CASIMIR caps the coupled rank at j <= 8 and the order at n <= 12; we publish j up to 10.
ORACLE_COUPLED_RANK_MAX = 8

#: Published labels above that cap, all of them at n = 11 or n = 12. Measured.
NO_ORACLE_LABEL_COUNT = 735

#: Of those, the ones carrying a coefficient above the noise floor, per site-type pair.
NO_ORACLE_NONZERO = {("O", "O"): 67, ("H", "O"): 122, ("H", "H"): 222}


def test_the_high_coupled_rank_sector_has_no_oracle_and_is_not_empty(parity_anisotropic):
    """735 published labels are outside CASIMIR's `j <= 8` cap, and they are not negligible.

    The largest `|C11|` of the whole H-H pair is one of them -- label `43c 54c 9`, 247.242 --
    so the missing oracle is not confined to the tail. This is a recorded gap, not a defect:
    `casimir` refuses `Dispersion 13` outright and prints nothing above `j = 8`, so these
    coefficients cannot be validated externally at all.
    """
    labels, coefficients = parity_anisotropic
    high = labels[:, 5] > ORACLE_COUPLED_RANK_MAX
    assert int(high.sum()) == NO_ORACLE_LABEL_COUNT
    assert set(labels[high][:, 0].tolist()) == {11, 12}
    for pair, row in ORACLE_ROW.items():
        scale = _order_scale(labels, coefficients, row)
        nonzero = sum(
            1 for (n, _, _, _, _, j), value in zip(labels.tolist(), coefficients[row])
            if j > ORACLE_COUPLED_RANK_MAX and abs(value) > NOISE_FRACTION * scale[n])
        assert nonzero == NO_ORACLE_NONZERO[pair], (pair, nonzero)
    row = ORACLE_ROW[("H", "H")]
    eleven = labels[:, 0] == 11
    leading = int(np.abs(coefficients[row][eleven]).argmax())
    assert labels[eleven][leading].tolist() == [11, 4, 5, 5, 7, 9]


# --------------------------------------------------------------------------------------
# Source-independence guard for this module specifically.
# --------------------------------------------------------------------------------------


def test_module_has_no_runtime_reference_dependency():
    """This module must not read reference data or shell out at runtime.

    Same guard as `test_atomic_polarizabilities.py`. Prose may name the reviewed source;
    executable dependencies on it are the thing being forbidden, so this checks for the
    mechanisms rather than for the word.
    """
    source = Path(inspect.getfile(test_module_has_no_runtime_reference_dependency)).read_text()

    # Exclude this guard's own body: it necessarily names the mechanisms it forbids.
    marker = "def test_module_has_no_runtime_reference_dependency"
    scanned = source.split(marker)[0]
    executable = "\n".join(
        line for line in scanned.splitlines() if not line.lstrip().startswith("#")
    )

    for mechanism in (
        "camcasp-reference",
        "import json",
        "import subprocess",
        "subprocess",
        "os.system",
        "os.popen",
        "check_output",
        "Popen",
        "importlib",
        "open(",
        "read_text",
        "loadtxt",
        "np.load",
        "pathlib.Path(",
    ):
        assert mechanism not in executable, f"forbidden runtime dependency: {mechanism}"

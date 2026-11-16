"""The exact grid tier preserves point/block decisions across OpenMP teams."""
import numpy as np
import pytest
import psi4

pytestmark = [pytest.mark.psi, pytest.mark.api, pytest.mark.quick]


@pytest.mark.parametrize("scheme", ["TREUTLER", "BECKE", "NAIVE", "STRATMANN", "SBECKE"])
@pytest.mark.parametrize("weightcut", [-1.0, 1e-15])
def test_exact_grid_decisions_across_threads(scheme, weightcut):
    # The optional tier deliberately retains a parallel-reduction split center
    # and allows its floating-point boundary decisions to differ.
    if hasattr(psi4.core.VBase, "initial_grid"):
        pytest.skip("Opt tier retains nondeterministic octree-center reduction.")
    mol = psi4.geometry("""
        O  -1.49  -0.04  0.00
        H  -1.85   0.84  0.00
        H  -0.53   0.07  0.00
        @O  1.42   0.04  0.00
        @H  1.77  -0.42  0.76
        @H  1.77  -0.42 -0.76
        no_com
        no_reorient
        symmetry c1
    """)
    psi4.set_options({
        "dft_nuclear_scheme": scheme, "dft_weights_tolerance": weightcut,
        "dft_radial_points": 50, "dft_spherical_points": 110,
        "dft_remove_distant_points": False, "dft_block_scheme": "OCTREE",
    })
    basis = psi4.core.BasisSet.build(mol, "ORBITAL", "def2-svp")
    old_threads = psi4.core.get_num_threads()
    grids = []
    try:
        for threads in (1, 4):
            psi4.set_num_threads(threads)
            grid = psi4.core.DFTGrid.build(mol, basis)
            points = []
            functions = []
            for block in grid.blocks():
                points.append(np.column_stack([
                    np.asarray(getattr(block, coordinate)()) for coordinate in ("x", "y", "z", "w")
                ]))
                functions.append(tuple(block.functions_local_to_global()))
            grids.append((np.vstack(points), functions))
    finally:
        psi4.set_num_threads(old_threads)
    assert grids[0][1] == grids[1][1]
    np.testing.assert_array_equal(grids[0][0], grids[1][0])

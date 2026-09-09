"""Irregular solid harmonics and pfit T functions against CamCASP's own solidh.

The oracle tables below are the bit patterns printed by CamCASP 6.0's
``src/pfit/shift.f90::solidh`` and the three lines of
``src/pfit/process.F90::T_functions`` that build a T row, compiled with
``gfortran -O0 -ffp-contract=off`` -- the same non-contracting arithmetic
libisapol is built with (see libisapol/CMakeLists.txt and SPEC.md 3.5.5).  With
``-O2 -march=native`` the Fortran contracts into FMAs and its own output moves
by up to 1.3e-13 relative, so the bitwise claim is against a non-contracting
build only.
"""
import struct

import numpy as np
import pytest

from psi4 import core

pytestmark = [pytest.mark.smoke]


def unhex(words):
    return np.array([struct.unpack(">d", bytes.fromhex(w))[0] for w in words])


def identical(a, b):
    return all(struct.pack("<d", x) == struct.pack("<d", y) for x, y in zip(a, b))


CAMCASP_SOLIDH = {
    (0.31, -1.7, 2.3): (
        "3FD63F2CCDA3BE31", "3FB8BAF446F3B85E", "3F8AAA6B9183746F", "BFB24766A3BF457D", "3F93BB6293D47176",
        "3F79ABD2418CF11D", "BFA198E0A4AD001E", "BF8925F13D77EE08", "BF72F96EE28990D3", "3F52BCA6E726180B",
        "3F6154C0F0719C16", "BF87C2A56AAC457B", "BF7F415DEF812DA7", "BF679509D2961D5F", "BF551C6E4EA60902",
        "3F61909E3A7C5D94", "BF537F7E40AE6B82", "3F40B2E1750C8C13", "BF66E4B9393E9F05", "BF6B4240F01F6E26",
        "BF54911937DF0D22", "BF4F0B82971B7028", "3F59D4859C4A54A1", "3F380A8C780A80AB", "3F35262441A3BB6C",
    ),
    (1.0, 0.0, 0.0): (
        "3FF0000000000000", "0000000000000000", "3FF0000000000000", "0000000000000000", "BFE0000000000000",
        "0000000000000000", "0000000000000000", "3FEBB67AE8584CAA", "0000000000000000", "8000000000000000",
        "BFE3988E1409212E", "0000000000000000", "0000000000000000", "0000000000000000", "3FE94C583ADA5B53",
        "0000000000000000", "3FD8000000000000", "8000000000000000", "0000000000000000", "BFE1E3779B97F4A8",
        "0000000000000000", "0000000000000000", "0000000000000000", "3FE7AA10D193C22D", "0000000000000000",
    ),
    (0.0, 0.0, -1.0): (
        "3FF0000000000000", "BFF0000000000000", "0000000000000000", "0000000000000000", "3FF0000000000000",
        "8000000000000000", "8000000000000000", "0000000000000000", "0000000000000000", "BFF0000000000000",
        "0000000000000000", "0000000000000000", "8000000000000000", "8000000000000000", "0000000000000000",
        "0000000000000000", "3FF0000000000000", "0000000000000000", "0000000000000000", "0000000000000000",
        "0000000000000000", "8000000000000000", "8000000000000000", "0000000000000000", "0000000000000000",
    ),
    (2.5, 2.5, 2.5): (
        "3FCD8F7208E6B82E", "3F9F87F11A8FB364", "3F9F87F11A8FB364", "3F9F87F11A8FB364", "BC30000000000000",
        "3F7D208A5A912E31", "3F7D208A5A912E31", "0000000000000000", "3F7D208A5A912E32", "BF51F009BAEA3645",
        "3F45F813333274B4", "3F45F813333274B4", "0000000000000000", "3F615E38C0355B50", "BF4C5CA64C764182",
        "3F4C5CA64C764182", "BF30BDE6F2B876EA", "BF1E40B162A045DF", "BF1E40B162A045DF", "0000000000000000",
        "3F3564532B0DA712", "BF3402A202D7BC52", "3F3402A202D7BC52", "BF2C4C7F3234D4E0", "0000000000000000",
    ),
    (-1.45365196, 0.0, -1.12168732): (
        "3FE16D9D80845534", "BFC731D63B4C45DB", "BFCE0F257BBB0259", "0000000000000000", "3F83C9AB69FCB430",
        "3FC152978CF6E1FF", "8000000000000000", "3FB67303B9552477", "8000000000000000", "3F9F350D3FB13643",
        "BFA2EA0A466C66A2", "0000000000000000", "BFB0B3A9B0644CE4", "0000000000000000", "BFA1AC3F9D2F4E89",
        "0000000000000000", "BF945F25DFD00182", "BF7D1617EA7CCB9F", "0000000000000000", "3F9BB926B08EC932",
        "0000000000000000", "3F9F1D4A5E4C41E7", "8000000000000000", "3F8C8333239BE4B2", "8000000000000000",
    ),
    (1.45365196, 0.0, -1.12168732): (
        "3FE16D9D80845534", "BFC731D63B4C45DB", "3FCE0F257BBB0259", "0000000000000000", "3F83C9AB69FCB430",
        "BFC152978CF6E1FF", "8000000000000000", "3FB67303B9552477", "0000000000000000", "3F9F350D3FB13643",
        "3FA2EA0A466C66A2", "0000000000000000", "BFB0B3A9B0644CE4", "8000000000000000", "3FA1AC3F9D2F4E89",
        "0000000000000000", "BF945F25DFD00182", "3F7D1617EA7CCB9F", "0000000000000000", "3F9BB926B08EC932",
        "0000000000000000", "BF9F1D4A5E4C41E7", "8000000000000000", "3F8C8333239BE4B2", "0000000000000000",
    ),
}
CAMCASP_WATER_T = (
    ("O", (0.0, 0.0, 3.5), (
        "3FD2492492492492", "3FB4E5E0A72F0539", "0000000000000000", "0000000000000000", "3F97E225515A4F1C",
        "0000000000000000", "0000000000000000", "0000000000000000", "0000000000000000", "3F7B4B985CF97EFB",
        "0000000000000000", "0000000000000000", "0000000000000000", "0000000000000000", "0000000000000000",
        "0000000000000000", "3F5F31D2B36647FA", "0000000000000000", "0000000000000000", "0000000000000000",
        "0000000000000000", "0000000000000000", "0000000000000000", "0000000000000000", "0000000000000000",
    )),
    ("O", (2.1, -1.3, 0.9), (
        "3FD858C470617B2E", "3FA95E585A6C3DAA", "3FBD98BC697E47F0", "BFB2525C414E2C89", "BF924659E8F6E95C",
        "3F9AB506ABA6B730", "BF908871D7F97D98", "3F9337C3C8B8FB03", "BFA349DA26A31287", "BF81B961B4DD9531",
        "BF715E98ABC2D47F", "3F65814F4E8FB1C3", "3F76634F018AA008", "BF8678612DC84FAE", "BF54B2F3B00DF5EE",
        "BF8C01738B2734C3", "BF02A593EBDD6140", "BF6EC22EBD5F3617", "3F630A7E753AF0BA", "BF349E2E590520DE",
        "3F44B1960C2BBC4E", "BF3C880B912C884F", "BF734D2A5D6265B7", "BF66A7FB3A1088BE", "BF6E063EF216FFAC",
    )),
    ("O", (-3.0, 2.0, -1.0), (
        "3FD11ACEE560242A", "BF938C5A2AB704C2", "BFAD528740128723", "3FA38C5A2AB704C2", "BF7EB7FB67B1E2E8",
        "3F7D058947D10F5C", "BF7359062FE0B4E8", "3F782F47BBD8E221", "BF8D058947D10F5C", "3F5D8597D804B8D1",
        "3F5A6277ECA90207", "BF5196FA9DC60159", "BF4EE6ED7B97B864", "3F628A8E7D5B083D", "3F46B552236B59A0",
        "3F6D0421D7ECB9A1", "3F272384D9BE8305", "BF42EC6F1D842004", "3F393B3ED2058006", "BF21D7604FD53B05",
        "3F3568D9F96646D4", "BF212A751F10454B", "BF45EF23E094CA51", "BF440FC29BA37474", "BF443AEAF74CA075",
    )),
    ("H1", (0.0, 0.0, 3.5), (
        "3FCA6B6509EA7831", "3FA4CEA8A53C5D93", "BF8A2D8AC2E062BB", "0000000000000000", "3F7F273C7D9E0A33",
        "BF71DAD5B3A5850E", "0000000000000000", "3F4676A12A9CEE67", "8000000000000000", "3F55FB5C80A3A0E0",
        "BF53650FBA1D2F2D", "0000000000000000", "3F33C79D0A119DF0", "8000000000000000", "BF045192F35C684C",
        "0000000000000000", "3F2CBD9247E32426", "BF32B8311C8789BF", "0000000000000000", "3F1A89758C9EC26C",
        "0000000000000000", "BEF52B4F39DB5583", "0000000000000000", "3EC2D5213AB26ACB", "8000000000000000",
    )),
    ("H1", (2.1, -1.3, 0.9), (
        "3FCDD5E7835863EB", "3F9A379C4F072C4F", "BFA70AB7E456147F", "3F90DBBF9526728B", "BF6150BEAF0E4E34",
        "BF8188E8C432FC14", "3F69A88BE6034D9F", "3F7AB282BA324A1B", "BF768CFB54FAD6B3", "BF55889338EEBD09",
        "BF25A66C7F417804", "3F0FAE230B142CDF", "3F5A3A85190013DB", "BF5627A0BB4F7D96", "BF4A02F85551DC29",
        "3F56C8A5C1A0093B", "BF25D306941EFD9C", "3F341D5F0ECE66B0", "BF1D6EFD5BFD485B", "3F209C9ECF036967",
        "BF1C10373EB911C6", "BF2E3CABD452BF00", "3F3A7C1E5A2D375D", "3F0AFECFE1A4A0B9", "BF33E57F91C88299",
    )),
    ("H1", (-3.0, 2.0, -1.0), (
        "3FD9494F9B2F2491", "3F7EBDD21211B30C", "3FB86A6290F86DEA", "BFBF940F4FFE171D", "BF9F5BFFE80AD259",
        "3F69B4C1712A1578", "BF709FACF48C56A9", "BF8B78F27395CBFA", "BFAA67E71D537454", "BF5CAEA615FCA4D8",
        "BF8274BA31614072", "3F87DED1EFD98846", "BF42ABB36F21C48A", "BF61F222C029A3F7", "BF921418AE247B41",
        "BF7EE3F69781C6DA", "3F6CE527D26FC25F", "BF45DC4BD3C96A74", "3F4C4614BDD09F73", "3F55C9A26613AB70",
        "3F74F1172EA34301", "BF4D13339D4B85C1", "BF38D70C06262814", "BF7959913C106B0B", "3F6C4997C34C83BF",
    )),
    ("H2", (0.0, 0.0, 3.5), (
        "3FCA6B6509EA7831", "3FA4CEA8A53C5D93", "BF8A2D8AC2E062BB", "0000000000000000", "3F7F273C7D9E0A33",
        "BF71DAD5B3A5850E", "0000000000000000", "3F4676A12A9CEE67", "8000000000000000", "3F55FB5C80A3A0E0",
        "BF53650FBA1D2F2D", "0000000000000000", "3F33C79D0A119DF0", "8000000000000000", "BF045192F35C684C",
        "0000000000000000", "3F2CBD9247E32426", "BF32B8311C8789BF", "0000000000000000", "3F1A89758C9EC26C",
        "0000000000000000", "BEF52B4F39DB5583", "0000000000000000", "3EC2D5213AB26ACB", "8000000000000000",
    )),
    ("H2", (2.1, -1.3, 0.9), (
        "3FD9B6A2A4048A25", "3FC0C86351170CA6", "3FA5765855CD8159", "BFB595638F3ADF2B", "3FA0424417B51F92",
        "3F9843448D7A734C", "BFA8665CC2B4FC07", "BF879F5FFF821AAE", "BF8F340755BF4F76", "3F69E78338953C30",
        "3F838212BD080B42", "BF939E4A59D8ECC2", "BF813CEA575F7E3E", "BF86C50EA359EBDC", "BF70745FBD2C6524",
        "3F48DA5B81C761AD", "BF60B34435F7A770", "3F6724DDEF54C494", "BF774657E1CB3074", "BF71CFF3591519D3",
        "BF778746F61EB324", "BF6C6A1B2F0FDD51", "3F457564FA9FBE8B", "BF2FDFEB96094FF1", "3F4C43D952526684",
    )),
    ("H2", (-3.0, 2.0, -1.0), (
        "3FCA35C3FF22C9C8", "3F511E0D2D5A559B", "BFA393DAB969379F", "3F919557F384A719", "BF718CF58F641EEF",
        "BF36254BFE42C6AD", "3F23E3D1888011E0", "3F74387FB1B73F3E", "BF76BFA1A7E9D254", "BF113514C210F470",
        "3F5009254A70F0E5", "BF3CCDFC4F0DF01A", "3F0D8777DD19E893", "BF109C4090FD43B3", "BF3B49D43FF8F9F2",
        "3F55B406B7942439", "3F2195958081F086", "3EF450611A037BB4", "BEE23EAA4A67650B", "BF216FCA0FDB1939",
        "3F239DD2D762BAE8", "BED7936A20D984C7", "3EF2C0228516AE9A", "BF0058560B9F39BC", "BF314E046F53146E",
    )),
)

WATER_FRAMES = {
    # H2O.axes: "z global Z x from H2 to H1" for H1 and the mirror for H2; the
    # oxygen carries no axes block, so it keeps the global frame.
    "O": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    "H1": [[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]],
    "H2": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
}
WATER_SITES = {
    "O": [0.0, 0.0, 0.0],
    "H1": [-1.45365196, 0.0, -1.12168732],
    "H2": [1.45365196, 0.0, -1.12168732],
}


def test_camcasp_solidh_is_reproduced_bitwise():
    for displacement, words in CAMCASP_SOLIDH.items():
        reference = unhex(words)
        assert reference.size == 25
        for rank in range(5):
            mine = core.isa_irregular_solid_harmonics(rank, list(displacement))
            assert len(mine) == (rank + 1) ** 2
            # solidh fills the same locations regardless of the requested rank.
            assert identical(mine, reference[: len(mine)]), (displacement, rank)


def test_reference_water_axes_give_bitwise_camcasp_t_functions():
    for site, point, words in CAMCASP_WATER_T:
        reference = unhex(words)
        mine = core.isa_t_functions(4, list(point), WATER_SITES[site], WATER_FRAMES[site])
        assert identical(mine, reference), (site, point)


def test_irregular_is_the_regular_harmonic_over_r_to_the_2k_plus_1():
    rng = np.random.default_rng(1729)
    worst = 0.0
    for _ in range(200):
        displacement = list(rng.normal(size=3) * rng.uniform(0.3, 4.0))
        radius = np.linalg.norm(displacement)
        irregular = np.array(core.isa_irregular_solid_harmonics(4, displacement))
        regular = np.array(core.isa_regular_multipoles(4, displacement))
        scale = np.concatenate([np.full(2 * k + 1, radius ** (-(2 * k + 1))) for k in range(5)])
        expected = regular * scale
        worst = max(worst, float(np.max(np.abs(irregular - expected) / np.abs(expected))))
    assert worst < 1.0e-11


def test_closed_forms_through_rank_two():
    x, y, z = 0.31, -1.7, 2.3
    r = np.sqrt(x * x + y * y + z * z)
    rt3 = np.sqrt(3.0)
    expected = [
        1 / r,
        z / r**3,
        x / r**3,
        y / r**3,
        (3 * z * z - r * r) / 2 / r**5,
        rt3 * x * z / r**5,
        rt3 * y * z / r**5,
        rt3 / 2 * (x * x - y * y) / r**5,
        rt3 * x * y / r**5,
    ]
    mine = core.isa_irregular_solid_harmonics(2, [x, y, z])
    assert np.allclose(mine, expected, rtol=0.0, atol=1.0e-15)


def test_rank_zero_is_one_over_r_and_not_one():
    displacement = [0.0, 0.0, 2.0]
    assert core.isa_irregular_solid_harmonics(0, displacement) == [0.5]
    assert core.isa_regular_multipoles(0, displacement) == [1.0]


def test_every_rank_is_harmonic_away_from_the_origin():
    displacement = np.array([0.31, -1.7, 2.3])
    step = 1.0e-4
    for rank in range(5):
        block = slice(rank * rank, (rank + 1) ** 2)
        centre = np.array(core.isa_irregular_solid_harmonics(rank, list(displacement)))[block]
        laplacian = -6.0 * centre
        for axis in range(3):
            for sign in (+1, -1):
                shifted = displacement.copy()
                shifted[axis] += sign * step
                laplacian = laplacian + np.array(
                    core.isa_irregular_solid_harmonics(rank, list(shifted)))[block]
        laplacian /= step * step
        assert np.max(np.abs(laplacian)) < 1.0e-6 * max(1.0, np.max(np.abs(centre)))


def test_singular_and_unsupported_requests_are_rejected():
    with pytest.raises(ValueError, match="singular at the origin"):
        core.isa_irregular_solid_harmonics(2, [0.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="rank must be in"):
        core.isa_irregular_solid_harmonics(5, [1.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="must be finite"):
        core.isa_irregular_solid_harmonics(1, [float("nan"), 0.0, 1.0])


def test_t_functions_are_the_irregular_harmonics_in_the_identity_frame():
    identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    point, site = [1.3, -0.7, 2.9], [0.2, 0.4, -0.5]
    mine = core.isa_t_functions(3, point, site, identity)
    direct = core.isa_irregular_solid_harmonics(3, [p - s for p, s in zip(point, site)])
    assert identical(mine, direct)


def test_t_functions_are_the_transposed_multipole_rotation_of_the_global_row():
    rng = np.random.default_rng(3)
    frame, upper = np.linalg.qr(rng.normal(size=(3, 3)))
    frame = frame * np.sign(np.diag(upper))
    if np.linalg.det(frame) < 0:
        frame[:, 0] *= -1
    columns = [list(row) for row in frame]
    rotation = np.array(core.isa_multipole_rotation(4, columns))
    point, site = list(rng.normal(size=3) * 2), list(rng.normal(size=3) * 0.5)
    mine = np.array(core.isa_t_functions(4, point, site, columns))
    global_row = np.array(core.isa_irregular_solid_harmonics(
        4, [p - s for p, s in zip(point, site)]))
    assert np.allclose(mine, rotation.T @ global_row, rtol=0.0, atol=1.0e-14)
    assert not np.allclose(mine, rotation @ global_row, rtol=0.0, atol=1.0e-6)


def test_t_functions_reject_frames_that_are_not_proper_rotations():
    reflection = [[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    with pytest.raises(ValueError, match="proper rotation"):
        core.isa_t_functions(1, [1.0, 0.0, 0.0], [0.0, 0.0, 0.0], reflection)
    skewed = [[1.0, 0.1, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    with pytest.raises(ValueError, match="orthogonal"):
        core.isa_t_functions(1, [1.0, 0.0, 0.0], [0.0, 0.0, 0.0], skewed)


def test_damping_is_the_tang_toennies_series_and_is_off_by_default():
    from math import exp, factorial
    for br in (0.0, 0.25, 1.3, 7.0, 40.0):
        for rank in range(5):
            series = sum(br**n / factorial(n) for n in range(rank + 2))
            assert core.isa_t_function_damping(rank, br) == pytest.approx(
                1.0 - exp(-br) * series, rel=0.0, abs=1.0e-15)
    assert core.isa_t_function_damping(4, 0.0) == 0.0
    assert core.isa_t_function_damping(0, 1.0e3) == 1.0
    identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    point, site = [1.3, -0.7, 2.9], [0.2, 0.4, -0.5]
    assert identical(core.isa_t_functions(4, point, site, identity),
                     core.isa_t_functions(4, point, site, identity, damping=0.0))


def test_damping_scales_each_rank_block_by_its_own_factor():
    identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    point, site, damping = [1.3, -0.7, 2.9], [0.2, 0.4, -0.5], 1.7
    undamped = np.array(core.isa_t_functions(4, point, site, identity))
    damped = np.array(core.isa_t_functions(4, point, site, identity, damping=damping))
    radius = np.linalg.norm([p - s for p, s in zip(point, site)])
    for rank in range(5):
        block = slice(rank * rank, (rank + 1) ** 2)
        factor = core.isa_t_function_damping(rank, damping * radius)
        assert np.allclose(damped[block], factor * undamped[block], rtol=0.0, atol=1.0e-16)
    live = undamped != 0.0
    assert live.sum() > 15
    assert np.all(np.abs(damped[live]) < np.abs(undamped[live]))


def test_damping_rejects_negative_reduced_distances():
    with pytest.raises(ValueError, match="nonnegative"):
        core.isa_t_function_damping(1, -1.0)
    identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    with pytest.raises(ValueError, match="nonnegative"):
        core.isa_t_functions(1, [1.0, 0.0, 0.0], [0.0, 0.0, 0.0], identity, damping=-0.5)

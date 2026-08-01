"""Dependency-free source guard for the native atomic-polarizability call path."""

import re


_PROCESS_CALL = re.compile(
    r"(?<![\w])(?:std\s*::\s*)?"
    r"(?:system|_?popen|v?fork|exec(?:l|le|lp|lpe|v|ve|vp|vpe)|posix_spawnp?|"
    r"CreateProcess[AW]?|ShellExecute[AW]?|WinExec)\s*\("
)
_EXTERNAL_TERM = re.compile(r"(?i)(?<![\w])(?:\.camcasp-reference|camcasp|orient|pfit|casimir)(?![\w])")
_INCLUDE_LINE = re.compile(r"^\s*#\s*include\b.*$", re.MULTILINE)


def _without_cpp_comments(text):
    """Remove C++ comments while preserving literals and line boundaries."""
    output = []
    index = 0
    mode = "code"

    while index < len(text):
        char = text[index]
        following = text[index + 1] if index + 1 < len(text) else ""

        if mode == "code":
            if char == "/" and following == "/":
                mode = "line-comment"
                index += 2
                continue
            if char == "/" and following == "*":
                mode = "block-comment"
                index += 2
                continue
            output.append(char)
            if char == '"':
                mode = "string"
            elif char == "'":
                mode = "character"
            index += 1
            continue

        if mode == "line-comment":
            if char == "\n":
                output.append(char)
                mode = "code"
            index += 1
            continue

        if mode == "block-comment":
            if char == "*" and following == "/":
                mode = "code"
                index += 2
            else:
                if char == "\n":
                    output.append(char)
                index += 1
            continue

        output.append(char)
        if char == "\\" and following:
            output.append(following)
            index += 2
            continue
        if (mode == "string" and char == '"') or (mode == "character" and char == "'"):
            mode = "code"
        index += 1

    return "".join(output)


def source_violations(text):
    """Return forbidden native launch calls and external/reference terms in active C++ text."""
    active_text = _INCLUDE_LINE.sub("", _without_cpp_comments(text))
    violations = []

    for match in _PROCESS_CALL.finditer(active_text):
        call = re.sub(r"\s+", "", match.group())
        violations.append(f"forbidden process API: {call}")
    for match in _EXTERNAL_TERM.finditer(active_text):
        violations.append(f"forbidden external term: {match.group().lower()}")

    return violations


def test_provider_uses_only_the_reviewed_native_response_route():
    source = (
        __import__("pathlib").Path(__file__).parents[2]
        / "psi4/src/psi4/libmints/atomic_polarizability.cc"
    ).read_text()
    body_start = source.index(
        "std::vector<SitePairResponse> ISAPolResponseProvider::compute_isapol_response"
    )
    body_end = source.index("\nPointResponseData::PointResponseData", body_start)
    body = _without_cpp_comments(source[body_start:body_end])

    for required in (
        "preflight_isapol_response_provider",
        "plan_isapol_response_provider",
        "construct_restricted_c1_primitives",
        "construct_restricted_alda_kernel",
        "assemble_restricted_singlet_hessian",
        "project_transition_multipoles",
        "solve_dense_restricted_response",
        "contract_site_pair_response",
    ):
        assert body.count(required) == 1
    assert body.index("preflight_isapol_response_provider") < body.index(
        "plan_isapol_response_provider"
    ) < body.index("construct_restricted_c1_primitives")
    plan_start = source.index("ISAPolResponsePlan plan_isapol_response_provider")
    plan_end = source.index(
        "SitePairResponseContractionPlan plan_site_pair_response_contraction",
        plan_start,
    )
    plan_body = _without_cpp_comments(source[plan_start:plan_end])
    for planner in (
        "plan_restricted_c1_jk",
        "plan_restricted_alda",
        "plan_transition_multipole_projection",
        "plan_site_pair_response_contraction",
    ):
        assert plan_body.count(planner) == 1
    assert "native point-response execution is not implemented" not in body
    assert "ao_multipole_potential" not in body
    assert "ExternalPotential" not in body
    assert source_violations(body) == []


def test_point_response_uses_only_canonical_native_construction_and_order_zero_potential():
    source = (
        __import__("pathlib").Path(__file__).parents[2]
        / "psi4/src/psi4/libmints/atomic_polarizability.cc"
    ).read_text()
    body_start = source.index("PointResponseData::PointResponseData")
    body_end = source.index("\nMatrix lw_graph_operator", body_start)
    body = _without_cpp_comments(source[body_start:body_end])

    assert body.count("ao_multipole_potential(0,") == 1
    assert "ao_multipoles" not in body
    assert "ExternalPotential" not in body
    assert "compute_isa" not in body.lower()
    assert "generate" not in body.lower()
    for required in (
        "plan_point_response_provider",
        "construct_restricted_c1_primitives",
        "construct_restricted_alda_kernel",
        "assemble_restricted_singlet_hessian",
        "solve_dense_restricted_response",
    ):
        assert required in body
    assert source_violations(body) == []

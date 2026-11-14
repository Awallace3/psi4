"""Experiment-only correction of delta-HF DiskDFJK namespace reuse.

Clone the Python driver in this process; never edit the staged installation.
This is unrelated to FDDS allocation and must be disclosed in resource results.
"""
import hashlib
import inspect


def install():
    from psi4.driver.procrouting.sapt import sapt_proc
    from psi4.driver.procrouting.proc_table import procedures

    original = sapt_proc.run_sapt_dft
    assert procedures["energy"]["sapt(dft)"] is original
    source = inspect.getsource(original)
    original_hash = hashlib.sha256(source.encode()).hexdigest()
    boundary = "    if hf_wfn_dimer is None:\n"
    assert source.count(boundary) == 1
    hf_source, dft_source = source.split(boundary)
    for start, end in [("dimer", "monomerA"), ("monomerA", "monomerB"), ("monomerB", "dimer")]:
        line = f"        core.IO.change_file_namespace(97, '{start}', '{end}')\n"
        # Restrict edits to the HF section; DFT already follows its moves
        # with set_default_namespace.
        if end == "dimer":
            line = "    " + line
        assert hf_source.count("\n" + line) == 1, (start, end)
        indent = line[:len(line) - len(line.lstrip())]
        hf_source = hf_source.replace("\n" + line, "\n" + line + f'{indent}core.IO.set_default_namespace("{end}")\n', 1)
    source = hf_source + boundary + dft_source
    namespace = dict(sapt_proc.__dict__)
    exec(compile(source, "<pr44-experiment-delta-hf-namespace>", "exec"), namespace)
    procedures["energy"]["sapt(dft)"] = namespace["run_sapt_dft"]
    return {
        "name": "delta-hf-file97-namespace-follow",
        "scope": "process-local Python driver only; no FDDS or native changes",
        "original_source_sha256": original_hash,
        "patched_source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "namespace_updates": 3,
    }

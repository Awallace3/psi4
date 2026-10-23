"""Reuse qualified pilot samples only across proven post-timing validator edits."""
import ast
import hashlib
import json
from pathlib import Path
import re
import subprocess

HERE=Path("devtools/disp0/phoenix_validation")
SHARED=("compare_cpu.py","cpu_payload.sh","cpu_job.sbatch","validate_cpu_allocation.py",
        "run_gate.py","gpu_runtime.py","fragment_gate.py","prepare_inputs.py")


def sha(data):
    return hashlib.sha256(data.encode() if isinstance(data,str) else data).hexdigest()


def node_key(node):
    return ast.dump(node,include_attributes=False)


def statements(text):
    return ast.parse(text).body


def qualification_key(main):
    """Normalize only this reviewed ledger replacement and diagnostic movement."""
    outer=next(n for n in main.body if isinstance(n,ast.Try))
    old=statements('''
np.testing.assert_allclose(sums["Disp"],canonical["Disp20,u"]+canonical["Exch-Disp20,u"],rtol=1e-8,atol=2e-8)
np.testing.assert_allclose(sums["Elst"],canonical["Elst10,r"],rtol=2e-5,atol=2e-7)
np.testing.assert_allclose(sums["Exch"],canonical["Exch10"],rtol=2e-5,atol=2e-7)
np.testing.assert_allclose(sum(sums.values()),float(energy),atol=2e-7,rtol=2e-5,err_msg="Full fragment sum vs total interaction")
''')
    ledger=statements('result["component_ledger"]=validate_component_ledger(sums,canonical,float(energy))')[0]
    snapshot=statements('np.savez(root/"matrices.npz",**arrays)')[0]
    observed=statements('''result.update(energy_Eh=float(energy),matrices_sha256=digest(root/"matrices.npz"),
signed_matrix_sums={name:float(np.sum(arrays[name])) for name in ("Elst","Exch","IndAB","IndBA","Disp")})''')[0]
    rank=statements('''rank=int(wfn.nmo())
result.update(retained_mo_rank=rank,retained_virtual_counts=[rank-o for o in counts])''')
    keys=[node_key(n) for n in outer.body]
    output=[];i=0;ledgers=snapshots=0;observations=rank_records=0
    while i<len(keys):
        if keys[i:i+4]==[node_key(n) for n in old]:
            output.extend(statements('qualification_component_ledger_marker()'))
            ledgers+=1;i+=4
        elif keys[i]==node_key(ledger):
            output.extend(statements('qualification_component_ledger_marker()'))
            ledgers+=1;i+=1
        elif keys[i]==node_key(snapshot):
            snapshots+=1;i+=1
        elif keys[i]==node_key(observed):
            observations+=1;i+=1
        elif keys[i:i+2]==[node_key(n) for n in rank]:
            rank_records+=1;i+=2
        else:
            output.append(outer.body[i]);i+=1
    if ledgers!=1 or snapshots!=1 or observations>1 or rank_records>1:
        raise RuntimeError("Unrecognized post-timing qualification; native pilot cannot be reused")
    outer.body=output
    return sha(node_key(main))


def timed_parts(text):
    tree=ast.parse(text)
    validators=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="validate_component_ledger"]
    for node in validators:
        args=(*node.args.posonlyargs,*node.args.args,*node.args.kwonlyargs)
        if (node.decorator_list or node.args.defaults or any(v is not None for v in node.args.kw_defaults)
                or node.returns is not None or any(a.annotation is not None for a in args)):
            raise RuntimeError("Post-timing validator may not have import-time side effects")
    main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="main")
    source=ast.get_source_segment(text,main)
    marker="\n        expected="
    if source.count(marker)!=1 or 'energy,wfn=psi4.energy(' not in source.split(marker)[0]:
        raise RuntimeError("Unrecognized timed CPU engine; pilot cannot be reused")
    # Every top-level statement except main and the declared post-call
    # validator must remain identical, including imports/options/timer class.
    other=[n for n in tree.body if not (isinstance(n,ast.FunctionDef)
                                      and n.name in ("main","validate_component_ledger"))]
    return {"main_through_energy_timer":sha(source.split(marker)[0]),
            "other_module_statements":sha(ast.dump(ast.Module(body=other,type_ignores=[]))),
            "all_other_qualification_rules":qualification_key(main)}


def submission_parts(text):
    tree=ast.parse(text)
    nodes=[n for n in tree.body if
           (isinstance(n,ast.FunctionDef) and n.name=="command") or
           (isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in
                                            ("SHAPES","CONSTRAINT") for t in n.targets))]
    start="    wall,mem,psi_mem,seconds=SHAPES[args.system]"
    end="    manifest="
    if text.count(start)!=1 or end not in text.split(start)[1]:
        raise RuntimeError("Unrecognized CPU submission environment")
    env=text.split(start)[1].split(end)[0]
    other=[n for n in tree.body if not (isinstance(n,ast.FunctionDef) and n.name=="main")]
    return {"resource_contract":sha(ast.dump(ast.Module(body=nodes,type_ignores=[]))),
            "other_submission_statements":sha(ast.dump(ast.Module(body=other,type_ignores=[]))),
            "worker_environment":sha(env)}


def revalidate_samples(pilot,expected,validate_ledger):
    import numpy as np
    from cpu_benchmark import production_options
    samples={}
    for repeat in range(3):
        for arm in ("cpp","einsums"):
            root=pilot/f"r{repeat}-{arm}"
            document=json.loads((root/"result.json").read_text())
            if document["status"]!="COMPLETED_CPU_REQUIRES_PAIRED_VALIDATION":
                raise RuntimeError("Original pilot child was not qualified")
            for key,value in {**expected,"system":"peptide","arm":arm,"repeat":repeat}.items():
                # Slurm records the short Phoenix hostname; os.uname()
                # records its FQDN. Accept only this explicit site suffix,
                # never arbitrary domains sharing a first DNS label.
                if key=="node" and "." not in value:
                    if document.get(key) in (value,value+".pace.gatech.edu"):
                        continue
                if document.get(key)!=value:
                    raise RuntimeError("Original pilot sample identity changed: "+key)
            options={**production_options(arm),"fisapt_fsapt_filepath":"none"}
            if document["options"]!=options:raise RuntimeError("Original pilot options changed")
            if sha((root/"matrices.npz").read_bytes())!=document["matrices_sha256"]:
                raise RuntimeError("Original pilot matrix artifact changed")
            with np.load(root/"matrices.npz",allow_pickle=False) as matrices:
                sums={name:float(np.sum(matrices[name])) for name in ("Elst","Exch","IndAB","IndBA","Disp")}
            for name,value in sums.items():
                np.testing.assert_allclose(value,document["signed_matrix_sums"][name],atol=1e-14,rtol=1e-12,
                                           err_msg="Pilot JSON sum vs hashed matrix "+name)
            ledger=validate_ledger(sums,document["canonical"],document["energy_Eh"])
            samples[root.name]={"result_sha256":sha((root/"result.json").read_bytes()),
                                "revalidated_component_ledger":ledger}
    return samples


def revalidate_pilot(source,pilot,old_pin,validate_ledger):
    if re.fullmatch(r"[0-9a-f]{40}",old_pin) is None:
        raise RuntimeError("Exact original pilot helper pin required")
    subprocess.run(["git","-C",str(source),"merge-base","--is-ancestor",old_pin,"HEAD"],check=True)
    def old(name):
        return subprocess.check_output(["git","-C",str(source),"show",old_pin+":"+str(HERE/name)],text=True)
    before=timed_parts(old("cpu_benchmark.py"))
    after=timed_parts((source/HERE/"cpu_benchmark.py").read_text())
    if before!=after:raise RuntimeError("Timed CPU engine changed; new native pilot required")
    if submission_parts(old("submit_cpu.py"))!=submission_parts((source/HERE/"submit_cpu.py").read_text()):
        raise RuntimeError("CPU resource/environment contract changed; new native pilot required")
    files={}
    for name in SHARED:
        left,right=sha(old(name)),sha((source/HERE/name).read_text())
        if left!=right:raise RuntimeError("Shared CPU runtime changed; new native pilot required: "+name)
        files[name]=left
    outcome=json.loads((pilot/"comparison-result.json").read_text())
    manifest=json.loads((pilot.parent/"manifest.json").read_text())
    allocation=json.loads((pilot/"allocation.json").read_text())
    config_path=Path(manifest["dependency_proof_config"])
    input_path=Path(manifest["input_manifest"])
    if (manifest["helper_pin"]!=old_pin or manifest["scientific_pin"]!=outcome["source_pin"]
            or manifest["input_sha256"]!=outcome["input_manifest_sha256"]
            or manifest["dependency_proof_sha256"]!=outcome["dependency_proof_sha256"]
            or sha(config_path.read_bytes())!=outcome["dependency_proof_sha256"]
            or sha(input_path.read_bytes())!=outcome["input_manifest_sha256"]
            or sha((pilot/"allocation.json").read_bytes())!=outcome["allocation_sha256"]
            or allocation["status"]!="PASS_CPU_ALLOCATION_ONLY"):
        raise RuntimeError("Original pilot manifest/allocation identity changed")
    config=json.loads(config_path.read_text())
    point=json.loads(input_path.read_text())["systems"]["peptide"]
    from validate_cpu_allocation import fields
    expected={"source_pin":outcome["source_pin"],
        "input_manifest_sha256":outcome["input_manifest_sha256"],
        "dependency_proof_sha256":outcome["dependency_proof_sha256"],
        "threads":manifest["cpus"],"psi4_memory":manifest["psi4_memory"],
        "geometry_sha256":point["geometry_sha256"],
        "frozen_occupied":[f["frozen_occupied"] for f in point["fragments"]],
        "stage":config["stage"],"node":fields(allocation["scheduler_node"])["NodeName"]}
    samples=revalidate_samples(pilot,expected,validate_ledger)
    from compare_cpu import compare
    paired={}
    for repeat in range(3):
        paired[str(repeat)]=compare(pilot/f"r{repeat}-cpp",pilot/f"r{repeat}-einsums")
        if repeat:
            for arm in ("cpp","einsums"):compare(pilot/f"r0-{arm}",pilot/f"r{repeat}-{arm}")
    return {"scope":"POST_TIMING_VALIDATOR_ONLY_REQUALIFICATION_NO_NEW_SCIENCE",
            "original_helper_pin":old_pin,"timed_contract":after,
            "shared_file_sha256":files,"samples":samples,"replayed_pair_comparisons":paired}

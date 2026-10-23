"""Freeze only the three requested literal geometries; no Psi4/CUDA execution."""
import argparse
import ast
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

SYSTEMS = ("peptide", "nanotube", "protein83")
Z = {"H":1,"C":6,"N":7,"O":8}
BASIS = {"primary":"aug-cc-pvdz","jkfit":"aug-cc-pvdz-jkfit","rifit":"aug-cc-pvdz-ri"}
EXPECTED = {"peptide":(24,356,1308,1042,(20,20)),
            "nanotube":(42,742,2748,2240,(8,78)),
            "protein83":(83,1293,4762,3820,(78,68))}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def literals(path):
    """Never import/execute a data module or select its off-limits systems."""
    source=path.read_bytes()
    tree=ast.parse(source,filename=str(path))
    assignments=[node for node in tree.body if isinstance(node,ast.Assign)
                 and any(isinstance(target,ast.Name) and target.id=="SYSTEMS" for target in node.targets)]
    if len(assignments)!=1:raise ValueError("One literal SYSTEMS dictionary required")
    values=ast.literal_eval(assignments[0].value)
    if not isinstance(values,dict) or any(not isinstance(values.get(name),str) for name in SYSTEMS):
        raise ValueError("All three requested literal geometries required")
    return {name:values[name] for name in SYSTEMS},digest(source)


def geometry_metadata(text):
    fragments=[Counter(),Counter()]
    side=0;header=False;commands=set()
    for original in text.splitlines():
        line=original.strip()
        if not line or line.startswith("#"):continue
        if line=="--":
            if side!=0 or not header or not fragments[0]:raise ValueError("Exactly two nonempty fragments required")
            side=1;header=False;continue
        if not header:
            if line!="0 1":raise ValueError("Only explicitly neutral closed-shell fragments admitted")
            header=True;continue
        if line in ("units angstrom","symmetry c1","no_reorient","no_com"):
            commands.add(line);continue
        fields=line.split()
        if len(fields)!=4 or fields[0] not in Z:
            raise ValueError("Only requested real H/C/N/O coordinates admitted")
        if any(not math.isfinite(float(v)) for v in fields[1:]):
            raise ValueError("Nonfinite coordinate")
        fragments[side][fields[0]]+=1
    if (side!=1 or not header or not all(fragments)
            or commands!={"units angstrom","symmetry c1","no_reorient","no_com"}):
        raise ValueError("Fixed two-fragment C1/Angstrom/no-reorient/no-com geometry required")
    result=[]
    for atoms in fragments:
        electrons=sum(Z[k]*n for k,n in atoms.items())
        if electrons%2:raise ValueError("Closed-shell electron counts required")
        result.append({"atoms":sum(atoms.values()),"composition":dict(sorted(atoms.items())),
            "electrons":electrons,"occupied":electrons//2,
            "frozen_occupied":sum(n for k,n in atoms.items() if k!="H")})
    return result


def spherical_counts(path):
    raw=path.read_bytes()
    lines=[line.strip() for line in raw.decode().splitlines()
           if line.strip() and not line.lstrip().startswith("!")]
    if lines[0].lower()!="spherical":raise ValueError("Spherical basis file required")
    counts={};max_l={};element=None
    angular={"S":0,"P":1,"D":2,"F":3,"G":4,"H":5,"I":6}
    for line in lines[1:]:
        fields=line.split()
        if line=="****":element=None;continue
        if element is None:
            if len(fields)!=2 or fields[1]!="0":raise ValueError("Unknown Gaussian element header")
            element=fields[0]
            counts[element]=0;max_l[element]=0;continue
        if len(fields)==3 and fields[0] in angular:
            l=angular[fields[0]]
            counts[element]+=2*l+1;max_l[element]=max(max_l[element],l)
        elif len(fields)==3 and fields[0]=="SP":
            counts[element]+=4;max_l[element]=max(max_l[element],1)
    if any(k not in counts or max_l[k]>3 for k in Z):
        raise ValueError("The requested H/C/N/O basis must remain SPDF")
    return {"filename":path.name,"sha256":digest(raw),
            "spherical_functions":{k:counts[k] for k in Z},
            "maximum_l":{k:max_l[k] for k in Z}}


def prepare(module,basis_directory,output):
    geometries,source_sha=literals(module)
    bases={role:spherical_counts(basis_directory/(name+".gbs")) for role,name in BASIS.items()}
    records={}
    for name,text in geometries.items():
        fragments=geometry_metadata(text)
        counts={role:sum(n*metadata["spherical_functions"][element]
                    for fragment in fragments for element,n in fragment["composition"].items())
                for role,metadata in bases.items()}
        occupied=tuple(f["occupied"] for f in fragments)
        actual=(sum(f["atoms"] for f in fragments),counts["primary"],
                counts["jkfit"],counts["rifit"],occupied)
        if actual!=EXPECTED[name]:raise ValueError("Requested dimensions changed: "+name)
        response=[]
        for f in fragments:
            o=f["occupied"];v=counts["primary"]-o;q=counts["jkfit"]
            response.append({"O":o,"V":v,"Q":q,"P":o*v,
                "resident_factor_bytes_only":8*q*(o*v+o*o+v*v)})
        records[name]={"geometry":text,"geometry_sha256":digest(text.encode()),
            "fragments":fragments,"spherical_basis_counts":counts,
            "response_resident_factor_lower_bounds":response}
    result={"schema":1,"scope":"FROZEN_INPUTS_AND_LOWER_BOUNDS_NOT_RUN_ADMISSION",
        "source_module":str(module),"source_module_sha256":source_sha,
        "systems":records,"basis_names":BASIS,"basis_files":bases,
        "warning":"Resident factors exclude native/HF plans, workspaces, IAO/IBO and solver buffers. "
                  "These estimates do not establish H200 memory admission or timing readiness."}
    output.mkdir(mode=0o700)
    with (output/"inputs.json").open("x") as stream:
        json.dump(result,stream,indent=2,sort_keys=True,allow_nan=False)
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--module",type=Path,required=True)
    parser.add_argument("--basis-directory",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    value=prepare(args.module.resolve(strict=True),args.basis_directory.resolve(strict=True),
                  args.output.resolve())
    print(json.dumps({"inputs":str(args.output.resolve()/"inputs.json"),
        "systems":{k:v["spherical_basis_counts"] for k,v in value["systems"].items()}},indent=2))


if __name__=="__main__":main()

#!/usr/bin/env python3
"""Validate exported native-DF inputs and replay their solve with NumPy.

This is not native Libint2 integral generation or end-to-end ISA certification.
Raw AO three-centre integrals are absent; the occupied-trace RHS is supplied.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np

spec=importlib.util.spec_from_file_location('native_df_reader',Path(__file__).with_name('replay_isa_checkpoint.py'))
base=importlib.util.module_from_spec(spec); spec.loader.exec_module(base)


def read_state(path):
    with Path(path).open() as stream:
        r=base.Reader(stream)
        r.expect('ISAPOL_NATIVE_DF 2')
        index,ndim,naux,nocc,ne=map(int,r.numbers(5,integer=True))
        if min(index,ndim,naux,nocc)<1 or nocc>ndim or ne!=2*nocc:
            raise ValueError('Invalid closed-shell dimensions')
        name,scf,runtime_scf=r.line(),r.line(),r.line()
        if not name or scf!='dalton' or runtime_scf!='dalton':
            raise ValueError('Unsupported SCF convention')
        route=r.line().split()
        if route not in [['Sc__','_A_A','T_MO','___A'],['Sc__','_B_B','T_MO','___B']]:
            raise ValueError('Unsupported constrained NN route')
        norm,solver,iterations,constraint=map(int,r.numbers(4,integer=True))
        if norm!=1 or solver!=0 or iterations<0 or constraint not in [1,2]:
            raise ValueError('Unsupported DF solver/norm/constraint')
        penalty=r.numbers(3)
        if not np.array_equal(penalty,[1000.,0.,0.]):
            raise ValueError('Unsupported penalty')
        r.expect('METRIC_COUNTS')
        metric_counts=r.numbers(2,integer=True)
        if np.any(metric_counts<1): raise ValueError('Missing metric records')
        bases,metadata={},{}
        for role,nf in [('MAIN',ndim),('AUX',naux)]:
            r.expect(role+'_METADATA')
            meta=dict(name=r.line(),role=r.line(),type=r.line())
            meta['charge']=int(r.numbers(1,integer=True)[0])
            basis=r.basis(role+'_BASIS')
            if basis['nfunction']!=nf or np.any(basis['shells'][:,1]>4):
                raise ValueError('Unsupported basis dimensions/angular momentum')
            bases[role],metadata[role]=basis,meta
        c=r.matrix('C_OCC',ndim,nocc)
        occupations=r.vector('OCCUPATIONS_ASSUMED_BY_CLOSED_SHELL_ROUTINE',nocc)
        if not np.array_equal(occupations,np.full(nocc,2.)):
            raise ValueError('Unsupported occupations')
        q=r.vector('Q',naux)
        rhs=r.vector('RHS_CONSTRAINED',naux)
        d=r.vector('DRHO',naux)
        r.expect('END_NATIVE_DF')
        if stream.read().strip(): raise ValueError('Trailing density data')
    return dict(index=index,ndim=ndim,naux=naux,nocc=nocc,nelectrons=ne,
                name=name,route=route,refinement_iterations=iterations,constraint=constraint,
                penalty=penalty,bases=bases,basis_metadata=metadata,c=c,
                occupations=occupations,q=q,rhs=rhs,d=d,metric_counts=metric_counts)


def read_metric(path,label,state):
    with Path(path).open() as stream:
        r=base.Reader(stream)
        r.expect('ISAPOL_NATIVE_METRIC 1')
        index=int(r.numbers(1,integer=True)[0])
        if index!=state['index']: raise ValueError('Metric molecule mismatch')
        r.expect(label)
        controls=r.numbers(2)
        if controls[0]<0 or controls[1]<=0: raise ValueError('Invalid integral controls')
        matrix=r.matrix('METRIC',state['naux'],state['naux'])
        r.expect('END_METRIC')
        if stream.read().strip(): raise ValueError('Trailing metric data')
    return matrix,controls


def replay(directory):
    directory=Path(directory)
    s=read_state(directory/'isapol-native-df-state.dat')
    metrics,controls,paths={},{},[directory/'isapol-native-df-state.dat']
    for label,count in zip(['J','A'],s['metric_counts']):
        expected=[directory/f'isapol-native-df-{label}-{i:06d}.dat' for i in range(1,int(count)+1)]
        if sorted(directory.glob(f'isapol-native-df-{label}*.dat'))!=expected:
            raise ValueError('Missing, extra or misnumbered metric records')
        records=[read_metric(p,label,s) for p in expected]
        metrics[label],controls[label]=records[0]
        if any(not np.array_equal(m,metrics[label]) or not np.array_equal(c,controls[label]) for m,c in records[1:]):
            raise ValueError('Repeated metric observations disagree; no record selected silently')
        paths.extend(expected)
    j,a=metrics['J'],metrics['A']
    jcontrols=controls['J']
    if not np.array_equal(jcontrols,controls['A']): raise ValueError('Integral controls changed')
    if np.any(np.diag(j)<=0): raise ValueError('Nonpositive Coulomb self metric')
    def error(x,y):
        absolute=float(np.max(np.abs(x-y)))
        return dict(max_absolute=absolute,max_scaled=absolute/max(1.,float(np.max(np.abs(y)))))
    reconstructed=j+s['penalty'][0]*np.outer(s['q'],s['q'])
    errors=dict(j_symmetry=error(j,j.T),a_symmetry=error(a,a.T),
                constrained_metric=error(a,reconstructed))
    solved=np.linalg.solve(a,s['rhs'])
    errors['numpy_coefficients']=error(solved,s['d'])
    residual=float(np.linalg.norm(a@s['d']-s['rhs'])/(np.linalg.norm(a)*np.linalg.norm(s['d'])+np.linalg.norm(s['rhs'])))
    tolerance=1e-9
    return dict(schema_version=1,evidence_class='exported-input DF equation checks and NumPy solve replay, not native generation',
        dimensions=dict(main=s['ndim'],aux=s['naux'],occupied=s['nocc']),
        metric_record_counts=s['metric_counts'].tolist(), repeated_metrics_identical=True,
        representation={k:v['representation'] for k,v in s['bases'].items()},
        basis_metadata=s['basis_metadata'],refinement_iterations=s['refinement_iterations'],
        integral_cutoff=float(jcontrols[0]),dummy_s_exponent=float(jcontrols[1]),
        errors=errors,reference_solve_residual=residual,condition_number_2=float(np.linalg.cond(a)),
        fitted_electrons=float(s['q']@s['d']),electron_count_error=float(s['q']@s['d']-s['nelectrons']),
        scaled_tolerance=tolerance,
        passed=bool(np.isfinite(residual) and residual<=tolerance and all(e['max_scaled']<=tolerance for e in errors.values())),
        limitations=['J/A, basis/C, q and occupied-trace RHS are exported, not independently generated',
                     'Raw AO three-centre B is absent',
                     'Occupation 2 is assumed by the reference density routine',
                     'Finite lambda charge penalty is not exact normalization; no coefficient rescaling',
                     'No native Libint2, ISA trajectory or downstream property certification'],
        source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    parser.add_argument('--report',type=Path,required=True)
    args=parser.parse_args()
    if args.report.exists(): raise FileExistsError(f'Refusing to overwrite {args.report}')
    report=replay(args.directory)
    args.report.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    if not report['passed']: raise SystemExit('DF equation/solve replay failed; evidence retained')

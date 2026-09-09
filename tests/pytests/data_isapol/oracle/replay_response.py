#!/usr/bin/env python3
"""Strict schema3 event association and exported internal-response equation replay.

This is a NumPy test oracle, not another production FDDS implementation. No native
integrals/SCF or shared-Psi4 kernel equivalence is established by this replay.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np

spec=importlib.util.spec_from_file_location('response_basis_reader',Path(__file__).with_name('replay_isa_checkpoint.py'))
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
spec=importlib.util.spec_from_file_location('response_cache_reader',Path(__file__).with_name('response_cache.py'))
cache=importlib.util.module_from_spec(spec);spec.loader.exec_module(cache)


def require(ok,message):
    if not ok: raise ValueError(message)


def fit_parameters(r):
    integers=tuple(map(int,r.numbers(4,integer=True)))
    values=r.vector('FIT_PARAMETERS',4)
    flags=tuple(map(int,r.numbers(2,integer=True)))
    require(integers[0]==1 and integers[3] in (0,1) and all(x in (0,1) for x in flags),'Unsupported fit metadata')
    return (integers,tuple(map(float,values)),flags)


def subset_parent_index(kind,row,o,v):
    count={'OO':o*(o+1)//2,'OV':o*v,'VV':v*(v+1)//2}[kind]
    require(1<=row<=count,'Subset row out of range')
    if kind=='OV':
        lower=(row-1)%o+1;upper=o+(row-1)//o+1
    else:
        upper=1
        while row>upper*(upper+1)//2:upper+=1
        lower=row-upper*(upper-1)//2
        if kind=='VV':lower+=o;upper+=o
    return lower+upper*(upper-1)//2


def read_event(path):
    with Path(path).open() as stream:
        r=base.Reader(stream)
        version=r.line()
        require(version in ('ISAPOL_RESPONSE_EVENT 3','ISAPOL_RESPONSE_EVENT 4','ISAPOL_RESPONSE_EVENT 5','ISAPOL_RESPONSE_EVENT 6','ISAPOL_RESPONSE_EVENT 7','ISAPOL_RESPONSE_EVENT 8'),'Unsupported response schema')
        header=tuple(map(int,r.numbers(7,integer=True)))
        serial,index,n,m,o,v,ne=header
        require(min(serial,index,n,m,o,v)>0 and ne==2*o and o+v<=n,'Invalid closed-shell dimensions')
        kind,name,scf=r.line(),r.line(),r.line()
        require(bool(name) and scf=='dalton','Unsupported molecule/convention')
        e=dict(header=header,kind=kind,name=name,scf=scf,version=int(version.split()[-1]))
        p=o*v
        if kind in cache.CACHE_KINDS:
            require(e['version']==8,'Cache lifecycle requires schema8')
            cache.read_fields(r,e,fit_parameters)
        elif kind=='HESSIAN_TENSOR':
            require(e['version']>=5,'Hessian tensors require schema5')
            e['hessian'],e['tensor_type'],e['description']=r.line(),r.line(),r.line()
            e['identity']=(r.line(),r.line())
            e['mapping']=tuple(map(int,r.numbers(8,integer=True)))
            require(e['hessian'] in ('h1','h2') and e['description'] in ('AAAA','BBBB'),'Invalid tensor consumer')
            require(e['mapping'][4:]==(1,0,1,0),'Unsupported tensor flags')
            if e['tensor_type']=='OVOV':
                require(e['mapping'][:4]==(2,2,o,o),'Invalid OVOV mapping')
                shape=(p,p)
            elif e['tensor_type']=='VVOO':
                require(e['mapping'][:4]==(1,1,0,0),'Invalid VVOO mapping')
                shape=(v*(v+1)//2,o*(o+1)//2)
            else:raise ValueError('Unknown Hessian tensor type')
            e['exchange']=float(r.vector('EXCHANGE',1)[0])
            e['matrix']=r.matrix('MATRIX',*shape)
        elif kind=='NUMERICAL_KERNEL_POLICY':
            require(e['version']>=5,'Numerical kernel policy requires schema5')
            e['branch']=r.line();e['batch']=int(r.numbers(1,integer=True)[0])
            e['cutoff']=float(r.vector('KERNEL_INTEGRAL_CUTOFF',1)[0])
            require(e['branch'] in ('ALDA','ALDAX') and e['batch']>0 and e['cutoff']>=0,'Invalid numerical kernel policy')
        elif kind=='SOLVE_END':
            require(e['version']>=7,'Parent solve requires schema7')
            e['solve_end']=tuple(map(int,r.numbers(2,integer=True)))
        elif (kind.startswith('FIT_') and kind!='FIT_METADATA') or kind=='SOLVE_BEGIN':
            require(e['version']>=4,'Raw fit events require schema4')
            if kind in ('FIT_BEGIN','SOLVE_BEGIN'):
                require(kind=='FIT_BEGIN' or e['version']>=7,'Parent solve requires schema7')
                e['fit_control']=tuple(map(int,r.numbers(6,integer=True)))
                fid,nmos,pairs,constraint,solver,iterations=e['fit_control']
                require(fid>0 and nmos==o+v and pairs==nmos*(nmos+1)//2,'Invalid NN fit counts')
                require(constraint in (1,2) and solver==iterations==0,'Unsupported NN solver policy')
                e['integrals']=tuple(r.line() for _ in range(4))
                valid=[('Sc__','_A_A','T_MO','___A'),('Sc__','_B_B','T_MO','___B')]
                if kind=='SOLVE_BEGIN':valid += [('S___','_A_A','T_MO','___A'),('S___','_B_B','T_MO','___B')]
                require(e['integrals'] in valid,'Invalid NN integral identities')
                for key in ('a_identity','b_identity','x_identity'):e[key]=(r.line(),r.line())
                e['shapes']=tuple(map(int,r.numbers(6,integer=True)))
                require(e['shapes']==(m,m,pairs,m,pairs,m),'Invalid NN original shapes')
                e['fit']=fit_parameters(r)
                lam=e['fit'][1][0]
                require(e['fit'][0][2]==1 and e['fit'][1][1:3]==(0.,0.) and
                        (lam==1. or (kind=='SOLVE_BEGIN' and lam==0.)),'Unsupported NN fit parameters')
                require(e['integrals'][0]==('Sc__' if lam==1. else 'S___'),'Parent metric/constraint mismatch')
            elif kind=='FIT_A':
                e['block_id']=tuple(map(int,r.numbers(2,integer=True)))
                e['matrix']=r.matrix('MATRIX',m,m)
            elif kind=='FIT_RHS_BLOCK':
                e['block_control']=tuple(map(int,r.numbers(7,integer=True)))
                fid,block,first,last,total,allocation,width=e['block_control']
                require(min(fid,block,first)>0 and first<=last<=total and width==last-first+1 and allocation>=width,'Invalid RHS block bounds')
                require((r.line(),r.line())==('N','T'),'Invalid RHS transpose policy')
                require(tuple(r.numbers(2,integer=True))==(total,m),'Invalid original RHS shape')
                e['matrix']=r.matrix('MATRIX',m,width)
            elif kind=='FIT_END':e['end_control']=tuple(map(int,r.numbers(5,integer=True)))
            else:raise ValueError(f'Unknown raw fit event {kind}')
        elif kind in ('J','H1','H2','KERNEL','CDF','KERNEL_OVOV_RAW'):
            e['identity']=(r.line(),r.line())
            e['extra']=r.vector('EXTRA',2)
            d=p if kind in ('H1','H2','KERNEL_OVOV_RAW') else m
            e['matrix']=r.matrix('MATRIX',d,d)
            if kind=='J': require(e['extra'][0]>=0 and e['extra'][1]>0,'Invalid integral controls')
            elif kind=='CDF': require(e['extra'][0]<=0 and e['extra'][1]==0,'Invalid imaginary frequency')
            elif kind=='KERNEL_OVOV_RAW':
                require(e['version']>=5 and 0<=e['extra'][0]<=1 and e['extra'][1] in (0,1),'Invalid raw-kernel controls')
            else: require(np.array_equal(e['extra'],[0.,0.]),'Unexpected matrix metadata')
        elif kind=='ORBITALS':
            e['c']=r.matrix('C',n,o+v);e['energies']=r.vector('ENERGIES',o+v)
        elif kind=='DIAGONAL_ENERGIES':
            e['fraction']=r.vector('DIAGONAL_FRACTION',1)
            e['energies']=r.vector('ENERGIES',o+v)
            e['bases']={k:r.basis(k+'_BASIS') for k in ('MAIN','AUX')}
            require(e['bases']['MAIN']['nfunction']==n and e['bases']['AUX']['nfunction']==m,'Basis dimensions')
            require(e['bases']['MAIN']['representation']=='S' and e['bases']['AUX']['representation']=='C','Basis representation')
        elif kind in ('OO_ROW','OV_ROW','VV_ROW'):
            require(kind=='OV_ROW' or e['version']>=6,'Paired subset rows require schema6')
            e['destination']=(r.line(),r.line());e['parent']=(r.line(),r.line())
            ar,ij=map(int,r.numbers(2,integer=True))
            require(ij==subset_parent_index(kind[:2],ar,o,v),'Packed/occupied-fast index mismatch')
            e.update(ar=ar,ij=ij,row=r.vector('ROW',m))
        elif kind=='KERNEL_DENSITY':
            require(e['version']>=6,'Kernel density requires schema6')
            e['density_identity']=(r.line(),r.line())
            e['density_name'],e['basis_role']=r.line(),r.line()
            e['identity']=(r.line(),r.line())
            require(tuple(r.numbers(3,integer=True))==(index,0,m),'Kernel density identity/dimensions')
            require(e['basis_role']=='AUX1' and bool(e['density_name']),'Kernel density basis role')
            e['coefficients']=r.vector('COEFFICIENTS',m)
        elif kind=='FIT_METADATA':
            e['identity']=(r.line(),r.line())
            e['shape']=tuple(map(int,r.numbers(2,integer=True)))
            require(min(e['shape'])>0 and e['shape'][1]==m,'Invalid fitted matrix shape')
            e['fit']=fit_parameters(r)
        elif kind=='KERNEL_SOURCE':
            e['identity']=(r.line(),r.line());e['density_identity']=(r.line(),r.line())
            e['constrained']=int(r.numbers(1,integer=True)[0])
            if e['version']>=6:
                e['density_serial']=int(r.numbers(1,integer=True)[0])
                require(0<e['density_serial']<serial,'Invalid kernel density generation')
        elif kind=='PROJECTION':
            e['constrained']=int(r.numbers(1,integer=True)[0])
            e['d_identity']=(r.line(),r.line());e['k_identity']=(r.line(),r.line())
            e['fit']=fit_parameters(r)
        elif kind=='POLICY':
            e['propagator'],e['hessians']=r.line(),r.line()
            e['controls']=tuple(map(int,r.numbers(6,integer=True)))
            solver,it,df,dfint,alda,constrained=e['controls']
            require(solver==0 and it>=0 and df==dfint==1 and alda in (0,1) and constrained in (0,1),'Unsupported response policy')
            require(e['propagator']=='cks' and e['hessians']=='internal','Unsupported response dispatch')
            e['exchange']=tuple(map(float,r.vector('EXCHANGE',2)))
        else: raise ValueError(f'Unknown event {kind}')
        if 'constrained' in e: require(e['constrained'] in (0,1),'Invalid constraint flag')
        for key in ('identity','destination','parent','d_identity','k_identity','density_identity','a_identity','b_identity','x_identity'):
            if key in e: require(all(e[key]),f'Empty {key}')
        r.expect('END_RESPONSE_EVENT')
        require(not stream.read().strip(),'Trailing event data')
    return e


def validate_fit_events(events):
    """Validate complete raw LU epochs and associate outgoing OV rows by parent."""
    completed=[];active=None;pending_a=None
    for e in events:
        kind=e['kind']
        if kind=='FIT_BEGIN':
            require(active is None,'Nested raw fit')
            fid,nmos,total,*_=e['fit_control']
            require(fid==len(completed)+1,'Nonsequential fit IDs')
            active=dict(begin=e,next=1,blocks=0,rhs=[],a=None)
        elif kind=='FIT_A':
            require(active is not None and pending_a is None,'Unexpected fit A')
            require(e['block_id']==(active['begin']['fit_control'][0],active['blocks']+1),'Fit A block identity mismatch')
            pending_a=e['matrix']
            if active['a'] is None:active['a']=pending_a
            require(np.array_equal(active['a'],pending_a),'Original A changed between LU blocks')
        elif kind=='FIT_RHS_BLOCK':
            require(active is not None and pending_a is not None,'RHS without pre-LU A')
            fid,block,first,last,total,allocation,width=e['block_control']
            require(fid==active['begin']['fit_control'][0] and block==active['blocks']+1,'RHS block identity mismatch')
            require(first==active['next'] and total==active['begin']['fit_control'][2],'Noncontiguous NN RHS')
            active['rhs'].append(e['matrix']);active['next']=last+1;active['blocks']+=1;pending_a=None
        elif kind=='FIT_END':
            require(active is not None and pending_a is None,'Unexpected fit end')
            fid,info,blocks,covered,total=e['end_control']
            require(fid==active['begin']['fit_control'][0] and info==0,'Failed/mismatched raw fit')
            require(blocks==active['blocks'] and blocks>0 and covered==total==active['begin']['fit_control'][2] and active['next']==total+1,'Incomplete NN RHS coverage')
            active['rhs']=np.concatenate(active['rhs'],axis=1);active['end']=e
            completed.append(active);active=None
    require(active is None and pending_a is None,'Unterminated raw fit')
    require(completed or not any(e.get('version',3)>=4 for e in events),'Schema4 missing raw NN fit')
    for i,f in enumerate(completed):
        parent=f['begin']['x_identity'];stop=min((g['begin']['header'][0] for g in completed[i+1:] if g['begin']['x_identity']==parent),default=len(events)+1)
        o,v=f['begin']['header'][4:6]
        rows=[e for e in events if e['kind']=='OV_ROW' and e['parent']==parent and f['end']['header'][0]<e['header'][0]<stop]
        require(len(rows)==o*v and [e['ar'] for e in rows]==list(range(1,o*v+1)),'Missing/ambiguous outgoing OV generation')
        require(len({e['destination'] for e in rows})==1,'Mixed OV output identity')
        f['ov']=np.asarray([e['row'] for e in rows])
        f['ov_indices']=np.asarray([e['ij']-1 for e in rows],dtype=int)
    return completed


def validate_subset_events(events, *, cache_state=None):
    """Chronological committed subsets; schema8 additionally requires cache lifecycles.

    Schema3-7 retain their prior independently incomplete cache status. Schema8 is
    an experimental reader contract until a fresh observer is validated.
    """
    if not any(e.get('version',3)>=6 for e in events):return None
    pending={};latest={};densities={};generations=[];used=[];projection=None;refreshes=[]
    parents={};solves=[];active_parent=None
    parent_required=any(e.get('version',3)>=7 for e in events)
    cache_required=any(e.get('version',3)>=8 for e in events)
    if cache_required and cache_state is None:cache_state=cache.CacheState()
    require(cache_required or cache_state is None,'Cache state requires schema8')
    for e in events:
        kind=e['kind'];serial=e['header'][0];o,v=e['header'][4:6];m=e['header'][3]
        if kind=='SOLVE_BEGIN':
            require(active_parent is None,'Nested parent solve')
            require(e['fit_control'][0]==len(solves)+1,'Nonsequential parent solve ID')
            require(not any(g['parent']==e['x_identity'] for g in pending.values()),'Parent replaced during subset production')
            parents.pop(e['x_identity'],None);active_parent=e
        elif kind=='SOLVE_END':
            require(active_parent is not None,'Parent completion without begin')
            require(e['solve_end']==(active_parent['fit_control'][0],0),'Failed/mismatched parent solve')
            g=dict(begin=active_parent,end=e)
            parents[active_parent['x_identity']]=g;solves.append(g);active_parent=None
        elif kind in ('OO_ROW','OV_ROW','VV_ROW'):
            dest=e['destination'];row=e['ar'];subset=kind[:2]
            require(e['ij']==subset_parent_index(subset,row,o,v),'Subset packed index mismatch')
            if row==1:
                require(dest not in pending,'Incomplete subset replaced before setter')
                latest.pop(dest,None)
                if parent_required:require(e['parent'] in parents,'Subset without successful parent solve')
                pending[dest]=dict(kind=subset,parent=e['parent'],parent_solve=parents.get(e['parent']),start=serial,data=[],
                    count={'OO':o*(o+1)//2,'OV':o*v,'VV':v*(v+1)//2}[subset])
            require(dest in pending,'Subset row without start')
            g=pending[dest]
            require(subset==g['kind'] and e['parent']==g['parent'] and row==len(g['data'])+1,'Nonsequential subset rows')
            require(row<=g['count'],'Excess subset rows')
            if parent_required:require(parents.get(e['parent']) is g['parent_solve'],'Subset parent changed')
            g['data'].append(e['row']);g['end']=serial
        elif kind=='FIT_METADATA':
            dest=e['identity']
            if dest in pending:
                g=pending.pop(dest)
                require(len(g['data'])==g['count'] and e['shape']==(g['count'],m),'Incomplete subset at setter')
                require(e['fit'][0][3]==1,'Uncommitted subset fit metadata')
                if parent_required:require(e['fit']==g['parent_solve']['begin']['fit'],'Subset/parent fit metadata mismatch')
                g.update(matrix=np.asarray(g.pop('data')),metadata=e,observed_metadata=e,identity=dest)
                latest[dest]=g;generations.append(g)
            elif dest in latest:
                g=latest[dest];old=g['observed_metadata'];before=old['fit'];after=e['fit']
                require(e['shape']==old['shape'],'Changed subset shape without producer')
                if before!=after:
                    # Pinned compare_type_df_parameters compares A%df_type to itself;
                    # do_DF_monomer still runs its final setter after skipping the fit.
                    # Only the observed NN->OV retag is supported here. A fresh direct
                    # OV solve is rejected by the observer's nrc_direct_guard.
                    retag=(g['kind']=='OV' and before[0][2]==1 and after[0][2]==3 and
                        before[0][:2]==after[0][:2] and before[0][3:]==after[0][3:] and before[1:]==after[1:])
                    require(retag,'Changed subset metadata without producer')
                    refreshes.append(dict(generation_start=g['start'],previous_serial=old['header'][0],
                        serial=serial,before=before,after=after,reason='cached NN-to-OV metadata retag'))
                # Preserve the successful producer's original metadata separately.
                g['observed_metadata']=e
        elif kind=='KERNEL_DENSITY':
            dest=e['density_identity']
            require(dest in latest and latest[dest]['kind']=='OO','Density without committed OO producer')
            g=latest[dest];values=np.zeros(m)
            for i in range(o):values+=g['matrix'][i+i*(i+1)//2,:]
            values*=2
            delta=float(np.max(np.abs(values-e['coefficients'])))
            densities[serial]=dict(event=e,source=g,max_absolute=delta,
                max_scaled=delta/max(1.,float(np.max(np.abs(e['coefficients'])))))
        elif kind=='KERNEL_SOURCE':
            sid=e.get('density_serial');require(sid in densities,'Unknown kernel density generation')
            d=densities[sid]
            require(e['density_identity']==d['event']['density_identity'],'Kernel density source linkage mismatch')
            require(bool(e['constrained'])==bool(d['source']['metadata']['fit'][1][0]>0),'Kernel density constraint selection mismatch')
            used.append(sid)
        elif kind=='PROJECTION':
            projection=e['d_identity']
            require(projection in latest and latest[projection]['kind']=='OV','Projection before subset setter')
        elif kind=='CDF':
            require(projection in latest and latest[projection]['kind']=='OV','Response before subset setter')
        if cache_state is not None:
            if kind=='DSD_REQUEST':require(active_parent is None,'Tensor during pending parent solve')
            cache_state.consume(e,latest)
    require(active_parent is None,'Unterminated parent solve')
    if parent_required:require(bool(solves),'Missing successful parent solves')
    require(not pending,'Subset stream missing successful setter')
    require(densities and set(densities)==set(used),'Unconsumed/missing kernel density')
    cache_report=cache_state.finish() if cache_state is not None else None
    return dict(generations=[dict(kind=g['kind'],identity=g['identity'],parent=g['parent'],
        start=g['start'],end=g['end'],setter=g['metadata']['header'][0],fit=g['metadata']['fit'],
        parent_solve_id=g['parent_solve']['begin']['fit_control'][0] if g['parent_solve'] else None) for g in generations],
        density_reconstruction=[dict(density_serial=sid,source_start=d['source']['start'],
            max_absolute=d['max_absolute'],max_scaled=d['max_scaled']) for sid,d in densities.items()],
        metadata_refreshes=refreshes,complete_parent_solve_provenance=parent_required,
        parent_solves=[dict(solve_id=s['begin']['fit_control'][0],identity=s['begin']['x_identity'],
            start=s['begin']['header'][0],end=s['end']['header'][0],fit=s['begin']['fit']) for s in solves],
        integral_cache_provenance=cache_report,
        cache_observer_validated=False,
        structural_cache_provenance_complete=cache_report is not None,
        complete_integral_cache_provenance=cache_report is not None,
        complete_parent_solve_and_cache_provenance=parent_required and cache_report is not None)


def associate(directory,expected_frequencies=11):
    require(expected_frequencies>=1,'Invalid expected frequency count')
    paths=sorted(Path(directory).glob('isapol-response-*.dat'))
    require(bool(paths),'No response events')
    events=[];identity=None
    rows={};pending={};fits={};kernels={};sources={};projection=None;policy=None;snapshots=[]
    for serial,path in enumerate(paths,1):
        require(path.name==f'isapol-response-{serial:06d}.dat','Missing/extra/misnumbered events')
        e=read_event(path);require(e['header'][0]==serial,'Serial mismatch')
        current=(e['header'][1:],e['name'],e['scf'],e['version'])
        if identity is None: identity=current
        require(current==identity,'Molecule/dimension/convention changed')
        events.append(e);kind=e['kind'];p=e['header'][4]*e['header'][5];m=e['header'][3]
        if kind=='OV_ROW':
            dest=e['destination'];ar=e['ar']
            if ar==1:
                require(dest not in pending,'Incomplete OV generation replaced')
                pending[dest]=dict(parent=e['parent'],start=serial,data=[])
                rows.pop(dest,None)  # A new generation invalidates the previous association.
            require(dest in pending,'OV row without producer start')
            group=pending[dest]
            require(e['parent']==group['parent'] and ar==len(group['data'])+1,'Nonsequential OV producer rows')
            group['data'].append(e['row'])
            if ar==p:
                rows[dest]=dict(matrix=np.asarray(group['data']),start=group['start'],end=serial,parent=group['parent'])
                del pending[dest]
        elif kind=='FIT_METADATA': fits[e['identity']]=e
        elif kind=='KERNEL_SOURCE': sources[e['identity']]=e
        elif kind=='KERNEL':
            require(e['identity'] in sources,'Kernel without source identity')
            kernels[e['identity']]=e
        elif kind=='PROJECTION':
            require(projection is None,'Multiple projection epochs unsupported in this reader')
            d,k=e['d_identity'],e['k_identity']
            require(d in rows and d in fits and k in kernels,'Missing producer for projection')
            require(fits[d]['shape']==(p,m) and fits[d]['fit']==e['fit'],'Projection fit metadata mismatch')
            projection=dict(event=e,d=rows[d],kernel=kernels[k],kernel_source=sources[k])
        elif kind=='POLICY':
            require(projection is not None,'Policy before projected kernel')
            require(e['controls'][-1]==projection['event']['constrained'],'Kernel/response D selectors differ')
            if policy is not None:
                require(e['controls']==policy['controls'] and e['exchange']==policy['exchange'],'Response policy changed')
            policy=e
        elif kind=='CDF':
            require(policy is not None and projection is not None,'CDF before response initialization')
            d=projection['event']['d_identity']
            require(d in rows and d in fits and d not in pending,'Missing current response-fit producer')
            require(fits[d]['shape']==(p,m),'Response-fit shape mismatch')
            require(sum(x['kind']=='POLICY' for x in events)==len(snapshots)+1,'Missing/repeated per-frequency policy')
            snapshots.append(dict(event=e,d=rows[d],fit=fits[d]['fit']))
    require(not pending,'Incomplete OV producer at end')
    for kind,count in [('ORBITALS',1),('DIAGONAL_ENERGIES',2),('H1',1),('H2',1),('PROJECTION',1),('CDF',expected_frequencies),('POLICY',expected_frequencies)]:
        require(sum(e['kind']==kind for e in events)==count,f'Unexpected {kind} event count')
    w=[float(x['event']['extra'][0]) for x in snapshots]
    require(w[0]==0 and all(x<0 for x in w[1:]) and len(set(w))==len(w),'Incomplete/duplicate frequency schedule')
    js=[e for e in events if e['kind']=='J'];require(bool(js),'Missing J')
    if events[0]['version']<8:
        require(all(np.array_equal(e['matrix'],js[0]['matrix']) and np.array_equal(e['extra'],js[0]['extra']) for e in js),'Repeated J differs')
    validate_fit_events(events)
    validate_subset_events(events)
    return events,snapshots,projection,policy,paths


def exchange_views(ovov,vvoo,o,v):
    """Literal occupied-fast access, without using physical tensor symmetry."""
    require(ovov.shape==(o*v,o*v) and vvoo.shape==(v*(v+1)//2,o*(o+1)//2),'Tensor dimensions')
    a=np.tile(np.arange(o),v);rr=np.repeat(np.arange(v),o)
    def packed(i,j):return np.maximum(i,j)*(np.maximum(i,j)+1)//2+np.minimum(i,j)
    x=ovov[rr[:,None]*o+a[None,:],rr[None,:]*o+a[:,None]]
    y=vvoo[packed(rr[:,None],rr[None,:]),packed(a[:,None],a[None,:])]
    return x,y


def reconstruct_hessians(events,projection,policy,error):
    if not any(e['version']>=5 for e in events):return None,None
    def unique(kind):
        found=[e for e in events if e['kind']==kind]
        require(len(found)==1,f'Expected one {kind}')
        return found[0]
    raw=unique('KERNEL_OVOV_RAW');numerical=unique('NUMERICAL_KERNEL_POLICY')
    h1=unique('H1');h2=unique('H2')
    tensors=[e for e in events if e['kind']=='HESSIAN_TENSOR']
    require(len(tensors)==4,'Incomplete Hessian tensor consumers')
    by={(e['hessian'],e['tensor_type']):e for e in tensors}
    require(set(by)=={(h,t) for h in ('h1','h2') for t in ('OVOV','VVOO')},'Repeated/missing tensor role')
    cx=policy['exchange'][0]
    require(policy['controls'][4]==0 and np.array_equal(raw['extra'],[cx,0.]),'Unsupported raw-kernel scaling policy')
    require(numerical['header'][0]<projection['kernel_source']['header'][0] and
            projection['event']['header'][0]<raw['header'][0]<h1['header'][0],'Kernel producer order')
    for e in tensors:
        require(e['exchange']==cx and e['header'][0]<(h1 if e['hessian']=='h1' else h2)['header'][0],'Tensor consumer order/exchange')
    if events[0]['version']<8:
        for t in ('OVOV','VVOO'):
            left,right=by['h1',t],by['h2',t]
            require(left['identity']==right['identity'] and left['description']==right['description'] and
                    np.array_equal(left['matrix'],right['matrix']),'Repeated Hessian tensor changed')
    require(by['h1','OVOV']['description']==by['h1','VVOO']['description'],'Mixed monomer tensors')
    o,v=h1['header'][4:6]
    energies=next(e['energies'] for e in events if e['kind']=='DIAGONAL_ENERGIES')
    delta=np.diag((energies[o:,None]-energies[:o]).ravel())
    vv=by['h1','OVOV']['matrix'];x,y=exchange_views(vv,by['h1','VVOO']['matrix'],o,v)
    # Preserve source addition order; do not symmetrize any input or result.
    calculated_h1=((4*(1-cx)*raw['matrix']+delta)+4*vv)-cx*x
    calculated_h1=calculated_h1-cx*y
    x2,y2=exchange_views(by['h2','OVOV']['matrix'],by['h2','VVOO']['matrix'],o,v)
    calculated_h2=(delta+cx*x2)-cx*y2
    d=projection['d']['matrix'];kernel=projection['kernel']['matrix']
    report=dict(h1_error=error(calculated_h1,h1['matrix']),h2_error=error(calculated_h2,h2['matrix']),
        auxiliary_projection_error=error(d@(kernel@d.T),raw['matrix']),
        numerical_kernel_branch=numerical['branch'],kernel_integral_cutoff=numerical['cutoff'],
        kernel_batch_size=numerical['batch'],local_kernel_multiplier=4*(1-cx))
    if events[0]['version']==8:
        state=cache.CacheState()
        provenance=validate_subset_events(events,cache_state=state)['integral_cache_provenance']
        report['integral_cache_provenance']=provenance
        first=next(u for u in state.consumers if u['event']['header'][0]==by['h1','OVOV']['header'][0])
        report['ordinary_fit_identity']=first['generation']['left']['identity']
        report['ordinary_D_J_D_error']=first['reconstruction']
        j=first['generation']['metric']['j']['matrix']
        report['response_D_J_D_difference']=error(d@(j@d.T),vv)
        report['limitations']=['Reader checks alone are not observer validation; require separate fresh-run evidence',
                              'No independent numerical-kernel quadrature reconstruction']
        return report,(calculated_h1,calculated_h2)
    # This is numerical agreement with a captured ordinary fit, not cache provenance.
    rows={};fits={}
    for e in events:
        if e['header'][0]>=by['h1','OVOV']['header'][0]:break
        if e['kind']=='OV_ROW':
            if e['ar']==1:rows[e['destination']]=[]
            rows[e['destination']].append(e['row'])
        elif e['kind']=='FIT_METADATA':fits[e['identity']]=e
    candidates=[(key,np.asarray(value)) for key,value in rows.items() if len(value)==o*v and key in fits and fits[key]['fit'][1]==(0.,0.,0.,0.)]
    require(len(candidates)==1,'Missing/ambiguous ordinary zero-penalty OV fit')
    identity,d0=candidates[0]
    # J may have repeated identical observations.
    j=next(e['matrix'] for e in events if e['kind']=='J')
    report['ordinary_fit_identity']=identity
    report['ordinary_D_J_D_error']=error(d0@(j@d0.T),vv)
    report['response_D_J_D_difference']=error(d@(j@d.T),vv)
    report['limitations']=['OO/VV coefficient rows and integral-cache generation metadata not exported',
                          'Numerical ordinary-D agreement is not complete cache provenance',
                          'No independent numerical-kernel quadrature reconstruction']
    return report,(calculated_h1,calculated_h2)


def replay(directory,expected_frequencies=11):
    events,snapshots,projection,policy,paths=associate(directory,expected_frequencies)
    def one(kind):return next(e for e in events if e['kind']==kind)
    def error(x,y):
        delta=float(np.max(np.abs(x-y)))
        return dict(max_absolute=delta,max_scaled=delta/max(1.,float(np.max(np.abs(y)))))
    h1,h2=one('H1')['matrix'],one('H2')['matrix']
    diagonals=[e for e in events if e['kind']=='DIAGONAL_ENERGIES']
    require(all(np.array_equal(e['fraction'],[1.]) for e in diagonals),'Unexpected diagonal fraction')
    require(np.array_equal(diagonals[0]['energies'],diagonals[1]['energies']),'H1/H2 energy inputs differ')
    for role in ('MAIN','AUX'):
        for key in diagonals[0]['bases'][role]:
            require(np.array_equal(diagonals[0]['bases'][role][key],diagonals[1]['bases'][role][key]),'H1/H2 basis changed')
    symmetry=dict(h1=error(h1,h1.T),h2=error(h2,h2.T))
    reconstruction,constructed=reconstruct_hessians(events,projection,policy,error)
    product=h2@h1;results=[];tolerance=1e-9
    for s in snapshots:
        d=s['d']['matrix'];reference=s['event']['matrix'];omega2=float(s['event']['extra'][0])
        a=product-omega2*np.eye(len(h1));rhs=-4*(h2@d)
        solution=np.linalg.solve(a,rhs);c=d.T@solution
        residual=float(np.linalg.norm(a@solution-rhs)/(np.linalg.norm(a)*np.linalg.norm(solution)+np.linalg.norm(rhs)))
        results.append(dict(omega2=omega2,cdf_error=error(c,reference),reference_reciprocity=error(reference,reference.T),
                            relative_backward_residual=residual,producer_start=s['d']['start'],producer_end=s['d']['end'],
                            fit_parameters=s['fit'],kernel_to_response_d_error=error(d,projection['d']['matrix'])))
        if constructed is not None:
            ch1,ch2=constructed
            cc=d.T@np.linalg.solve(ch2@ch1-omega2*np.eye(len(h1)),-4*(ch2@d))
            results[-1]['reconstructed_hessian_cdf_error']=error(cc,reference)
    raw_fits=[]
    for f in validate_fit_events(events):
        solution=np.linalg.solve(f['a'],f['rhs'])
        raw_fits.append(dict(fit_id=f['begin']['fit_control'][0],blocks=f['blocks'],
            ov_error=error(solution[:,f['ov_indices']].T,f['ov']),
            relative_backward_residual=float(np.linalg.norm(f['a']@solution-f['rhs'])/
                (np.linalg.norm(f['a'])*np.linalg.norm(solution)+np.linalg.norm(f['rhs'])))))
    passed=all(x['ov_error']['max_scaled']<=tolerance and np.isfinite(x['relative_backward_residual']) and
               x['relative_backward_residual']<=tolerance for x in raw_fits) and all(x['max_scaled']<=tolerance for x in symmetry.values()) and all(
        x['cdf_error']['max_scaled']<=tolerance and x['reference_reciprocity']['max_scaled']<=tolerance and
        np.isfinite(x['relative_backward_residual']) and x['relative_backward_residual']<=tolerance for x in results)
    if reconstruction is not None:
        passed=passed and all(reconstruction[k]['max_scaled']<=tolerance for k in
            ('h1_error','h2_error','auxiliary_projection_error','ordinary_D_J_D_error')) and all(
            x['reconstructed_hessian_cdf_error']['max_scaled']<=tolerance for x in results)
    subsets=validate_subset_events(events)
    if subsets is not None:
        passed=passed and all(d['max_scaled']<=tolerance for d in subsets['density_reconstruction'])
        if subsets['integral_cache_provenance'] is not None:
            passed=passed and subsets['integral_cache_provenance']['numerical_passed']
    return dict(schema_version=1,subset_density_provenance=subsets,hessian_reconstruction=reconstruction,evidence_class='exported old/internal response equation replay; not native generation',
        passed=bool(passed),scaled_tolerance=tolerance,dimensions=one('H1')['header'][2:6],symmetry=symmetry,
        policy=dict(controls=policy['controls'],exchange=policy['exchange']),frequencies=results,raw_fits=raw_fits,
        producer_to_diagonal_energy_error=error(one('ORBITALS')['energies'],diagonals[0]['energies']),
        source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        reader_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        cache_reader_sha256=hashlib.sha256(Path(cache.__file__).read_bytes()).hexdigest(),
        limitations=['H1/H2/D are exported, not generated by the Psi4 FDDS provider',
                     'Response D selection follows source-verified old use_constraints selector and temporal file producers',
                     'No raw OVOV/exchange-integral or quadrature-weight export',
                     'No shared-kernel equivalence, native SCF, ISA or end-to-end certification'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path);parser.add_argument('--report',type=Path,required=True)
    args=parser.parse_args()
    if args.report.exists():raise FileExistsError(args.report)
    report=replay(args.directory)
    with args.report.open('x') as f:json.dump(report,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in report.items() if k!='source_sha256'},indent=2))
    if not report['passed']:raise SystemExit('Internal response replay failed; report retained')

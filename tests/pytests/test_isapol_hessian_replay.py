"""Raw tensor permutations and schema5 Hessian reconstruction, without symmetry shortcuts."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest

spec=importlib.util.spec_from_file_location('hessian_fixtures',Path(__file__).with_name('test_isapol_response_replay.py'))
f=importlib.util.module_from_spec(spec);spec.loader.exec_module(f)
reader=f.reader


def test_exchange_views_nonsymmetric():
    o,v=2,3;p=o*v
    ov=np.arange(p*p,dtype=float).reshape(p,p)
    vv=np.arange(18,dtype=float).reshape(6,3)+100
    x,y=reader.exchange_views(ov,vv,o,v)
    def u(i,j):return min(i,j)+max(i,j)*(max(i,j)+1)//2
    for a in range(o):
        for r in range(v):
            for b in range(o):
                for s in range(v):
                    assert x[a+o*r,b+o*s]==ov[b+o*r,a+o*s]
                    assert y[a+o*r,b+o*s]==vv[u(r,s),u(a,b)]


def hessian_fixture():
    events=f.raw_fixture();d=np.array([[.2],[.3],[-.1],[.25]]);d0=d-.02
    ov=d0@d0.T;vv=np.outer([.11,.21,.13],[-.2,.15,.3]);kernel=.5*d@d.T
    energy=np.array([-1.,-.7,.2,1.]);delta=np.diag((energy[2:,None]-energy[:2]).ravel())
    h1=3*kernel+delta;h2=delta.copy()
    def u(i,j):return min(i,j)+max(i,j)*(max(i,j)+1)//2
    for a in range(2):
        for r in range(2):
            for b in range(2):
                for s in range(2):
                    i,j=a+2*r,b+2*s
                    h1[i,j]=h1[i,j]+4*ov[i,j]-.25*ov[b+2*r,a+2*s]
                    h1[i,j]-=.25*vv[u(r,s),u(a,b)]
                    h2[i,j]=h2[i,j]+.25*ov[b+2*r,a+2*s]-.25*vv[u(r,s),u(a,b)]
    result=[];frequency=0
    for kind,body in events:
        if kind=='KERNEL_SOURCE':
            for ar,row in enumerate(d0,1):
                a=(ar-1)%2+1;r=2+(ar-1)//2+1
                result.append(('OV_ROW','Ordinary OV\nDov0\nFull zero\nDfull0\n'+f'{ar} {a+r*(r-1)//2}\n'+f.vector('ROW',row)))
            result.append(('FIT_METADATA','Ordinary OV\nDov0\n4 1\n1 0 1 1\n'+f.vector('FIT_PARAMETERS',[0,0,0,0])+'0 0\n'))
            result.append(('NUMERICAL_KERNEL_POLICY','ALDA\n128\n'+f.vector('KERNEL_INTEGRAL_CUTOFF',[1e-12])))
            body=body.replace('Dooc\n1\n','Doo0\n0\n')
        if kind in ('H1','H2'):
            for typ,values,mapping in [('OVOV',ov,'2 2 2 2'),('VVOO',vv,'1 1 0 0')]:
                result.append(('HESSIAN_TENSOR',f'{kind.lower()}\n{typ}\nAAAA\n{typ} tensor\n{typ}file\n'+
                               mapping+' 1 0 1 0\n'+f.vector('EXCHANGE',[.25])+f.matrix('MATRIX',values)))
            body=f'{kind} name\n{kind}file\n'+f.vector('EXTRA',[0,0])+f.matrix('MATRIX',h1 if kind=='H1' else h2)
        elif kind=='POLICY':body=body.replace('0 0 1 1 1 1\n','0 0 1 1 0 1\n')
        elif kind=='CDF':
            w=[0.,-1.][frequency];frequency+=1
            c=d.T@np.linalg.solve(h2@h1-w*np.eye(4),-4*(h2@d))
            body='Response\nCDFfile\n'+f.vector('EXTRA',[w,0])+f.matrix('MATRIX',c)
        result.append((kind,body))
        if kind=='PROJECTION':
            result.append(('KERNEL_OVOV_RAW','Raw projection\nKraw\n'+f.vector('EXTRA',[.25,0])+f.matrix('MATRIX',kernel)))
    return result


def test_schema5_reconstructs_hessians_and_response(tmp_path):
    f.write_events(tmp_path,hessian_fixture(),version=5)
    r=reader.replay(tmp_path,expected_frequencies=2)
    assert r['passed']
    assert r['hessian_reconstruction']['ordinary_D_J_D_error']['max_scaled']<1e-14
    # The different lambda1 fit must not be silently used for Coulomb construction.
    assert r['hessian_reconstruction']['response_D_J_D_difference']['max_scaled']>1e-3


@pytest.mark.parametrize('fault',['mapping','duplicate_role','missing_tensor','changed_repeat',
                                  'numerical_branch','future_tensor','missing_ordinary'])
def test_schema5_rejects_bad_hessian_association(tmp_path,fault):
    events=hessian_fixture()
    if fault=='missing_tensor':events.pop(next(i for i,e in enumerate(events) if e[0]=='HESSIAN_TENSOR'))
    elif fault=='future_tensor':events.append(events.pop(next(i for i,e in enumerate(events) if e[0]=='HESSIAN_TENSOR')))
    elif fault=='missing_ordinary':events=[e for e in events if not(e[0] in ('OV_ROW','FIT_METADATA') and 'Dov0\n' in e[1])]
    else:
        for i,(kind,body) in enumerate(events):
            if fault=='mapping' and kind=='HESSIAN_TENSOR':events[i]=(kind,body.replace('2 2 2 2 1','2 2 3 3 1'))
            elif fault=='duplicate_role' and kind=='HESSIAN_TENSOR' and body.startswith('h2\nVVOO'):
                events[i]=(kind,body.replace('h2\n','h1\n',1))
            elif fault=='changed_repeat' and kind=='HESSIAN_TENSOR' and body.startswith('h2\nOVOV'):
                before=body.split('MATRIX\n')[0]
                events[i]=(kind,before+f.matrix('MATRIX',np.eye(4)))
            elif fault=='numerical_branch' and kind=='NUMERICAL_KERNEL_POLICY':events[i]=(kind,body.replace('ALDA','CKS'))
    f.write_events(tmp_path,events,version=5)
    with pytest.raises(ValueError):reader.replay(tmp_path,expected_frequencies=2)

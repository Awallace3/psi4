"""Synthetic schema3 response replay and temporal-association tests."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest

spec=importlib.util.spec_from_file_location('response_replay_test',Path(__file__).parent/'data_isapol/oracle/replay_response.py')
reader=importlib.util.module_from_spec(spec);spec.loader.exec_module(reader)


def vector(label,x):
    x=np.asarray(x).ravel()
    return f'{label}\n{len(x)}\n'+ ' '.join(format(float(v),'.17g') for v in x)+'\n'


def matrix(label,x):
    x=np.asarray(x)
    return f'{label}\n{x.shape[0]} {x.shape[1]}\n'+''.join(' '.join(format(float(v),'.17g') for v in row)+'\n' for row in x)


def basis(label,n,representation):
    return (f'{label}\n{n} {n} 1 {n} 1\n{representation}\nX\n4 0 0 0\n'+
            ''.join(f'{i+1} 1\n' for i in range(n))+
            ''.join(f'1 0 {i+1} {i+1}\n' for i in range(n)))


def fixture_events(change=False):
    e=np.array([-1.,-.7,.2,1.]);gap=(e[2:,None]-e[:2]).ravel()
    d=np.array([[.2],[.3],[-.1],[.25]])
    h2=np.diag(gap);h1=h2+6*d@d.T
    fit='1 0 1 1\n'+vector('FIT_PARAMETERS',[1,0,0,0])+'0 1\n'
    identity='OV coefficients\nDovc\n'
    ev=[]
    def add(kind,body):ev.append((kind,body))
    def rows(values):
        for ar,row in enumerate(values,1):
            a=(ar-1)%2+1;r=2+(ar-1)//2+1
            add('OV_ROW',identity+'Full coefficients\nDfull\n'+f'{ar} {a+r*(r-1)//2}\n'+vector('ROW',row))
        add('FIT_METADATA',identity+'4 1\n'+fit)
    def mat(kind,identity,value,extra=(0,0)):
        add(kind,identity+vector('EXTRA',extra)+matrix('MATRIX',value))
    add('ORBITALS',matrix('C',np.eye(4))+vector('ENERGIES',e))
    mat('J','Coulomb\nJfile\n',[[1]],(1e-12,1e-18))
    rows(d)
    add('KERNEL_SOURCE','Kernel\nKfile\nDensity fit\nDooc\n1\n')
    mat('KERNEL','Kernel\nKfile\n',[[.5]])
    add('PROJECTION','1\n'+identity+'Kernel\nKfile\n'+fit)
    diag=vector('DIAGONAL_FRACTION',[1])+vector('ENERGIES',e)+basis('MAIN_BASIS',4,'S')+basis('AUX_BASIS',1,'C')
    add('DIAGONAL_ENERGIES',diag);mat('H1','Electric\nH1file\n',h1)
    add('DIAGONAL_ENERGIES',diag);mat('H2','Magnetic\nH2file\n',h2)
    for i,w in enumerate([0.,-1.]):
        if i and change:
            d=d*.8;rows(d)
        add('POLICY','cks\ninternal\n0 0 1 1 1 1\n'+vector('EXCHANGE',[.25,.25]))
        c=d.T@np.linalg.solve(h2@h1-w*np.eye(4),-4*h2@d)
        mat('CDF','Response\nCDFfile\n',c,(w,0))
    return ev


def write_events(path,events,version=3):
    for i,(kind,body) in enumerate(events,1):
        (path/f'isapol-response-{i:06d}.dat').write_text(
            f'ISAPOL_RESPONSE_EVENT {version}\n{i} 1 4 1 2 2 4\n{kind}\nX\ndalton\n'+body+'END_RESPONSE_EVENT\n')


@pytest.mark.parametrize('change',[False,True])
def test_response_equation_replay_and_temporal_snapshots(tmp_path,change):
    write_events(tmp_path,fixture_events(change))
    report=reader.replay(tmp_path,expected_frequencies=2)
    assert report['passed']
    errors=[x['kernel_to_response_d_error']['max_absolute'] for x in report['frequencies']]
    assert errors[0]==0
    assert (errors[1]>0)==change
    starts=[x['producer_start'] for x in report['frequencies']]
    assert (starts[0]!=starts[1])==change


@pytest.mark.parametrize('fault',['trailing','nan','unknown','packed','missing_file','extra_file',
                                  'missing_producer','metadata','duplicate_frequency','incomplete_rows'])
def test_response_reader_rejects_corruption(tmp_path,fault):
    events=fixture_events()
    if fault=='unknown':events[0]=('UNKNOWN',events[0][1])
    elif fault=='packed':
        i=next(i for i,x in enumerate(events) if x[0]=='OV_ROW')
        events[i]=(events[i][0],events[i][1].replace('1 4\n','1 99\n'))
    elif fault=='missing_producer':events=[x for x in events if x[0]!='OV_ROW']
    elif fault=='metadata':
        i=next(i for i,x in enumerate(events) if x[0]=='PROJECTION')
        events[i]=(events[i][0],events[i][1].replace('Dovc\n','Missing\n'))
    elif fault=='duplicate_frequency':
        i=max(i for i,x in enumerate(events) if x[0]=='CDF')
        events[i]=(events[i][0],events[i][1].replace('-1 0\n','0 0\n'))
    elif fault=='incomplete_rows':events.append(next(x for x in events if x[0]=='OV_ROW'))
    write_events(tmp_path,events)
    first=tmp_path/'isapol-response-000001.dat'
    if fault=='trailing':first.write_text(first.read_text()+'unexpected\n')
    elif fault=='nan':first.write_text(first.read_text().replace('1 0 0 0\n','nan 0 0 0\n',1))
    elif fault=='missing_file':first.unlink()
    elif fault=='extra_file':(tmp_path/'isapol-response-extra.dat').write_text(first.read_text())
    with pytest.raises(ValueError):reader.replay(tmp_path,expected_frequencies=2)


def raw_fixture():
    events=fixture_events()
    fit='1 0 1 1\n'+vector('FIT_PARAMETERS',[1,0,0,0])+'0 1\n'
    begin=('1 4 10 1 0 0\nSc__\n_A_A\nT_MO\n___A\n'
           'Constrained metric\nScfile\nNN RHS\nTfile\nFull coefficients\nDfull\n'
           '1 1 10 1 10 1\n'+fit)
    rhs=np.zeros((1,10));rhs[0,[3,4,6,7]]=[.4,.6,-.2,.5]
    raw=[('FIT_BEGIN',begin)]
    for block,(first,last) in enumerate([(1,6),(7,10)],1):
        raw.append(('FIT_A',f'1 {block}\n'+matrix('MATRIX',[[2.]])))
        raw.append(('FIT_RHS_BLOCK',f'1 {block} {first} {last} 10 6 {last-first+1}\nN\nT\n10 1\n'+
                    matrix('MATRIX',rhs[:,first-1:last])))
    raw.append(('FIT_END','1 0 2 10 10\n'))
    return events[:2]+raw+events[2:]


def test_raw_nn_fit_multiblock_replay(tmp_path):
    write_events(tmp_path,raw_fixture(),version=4)
    report=reader.replay(tmp_path,expected_frequencies=2)
    assert report['passed'] and len(report['raw_fits'])==1
    assert report['raw_fits'][0]['blocks']==2
    assert report['raw_fits'][0]['ov_error']['max_absolute']==0


@pytest.mark.parametrize('fault',['gap','failed_end','unmatched_parent','changed_A',
                                  'missing_end','allocation_small','raw_in_schema3'])
def test_raw_nn_fit_rejects_corrupt_epochs(tmp_path,fault):
    events=raw_fixture()
    if fault=='missing_end':events=[e for e in events if e[0]!='FIT_END']
    else:
        for i,(kind,body) in enumerate(events):
            if fault=='gap' and kind=='FIT_RHS_BLOCK' and body.startswith('1 2 '):
                events[i]=(kind,body.replace('1 2 7 10','1 2 8 11'))
            elif fault=='failed_end' and kind=='FIT_END':events[i]=(kind,'1 -2 2 10 10\n')
            elif fault=='unmatched_parent' and kind=='FIT_BEGIN':events[i]=(kind,body.replace('Dfull','Missing'))
            elif fault=='changed_A' and kind=='FIT_A' and body.startswith('1 2\n'):
                events[i]=(kind,'1 2\n'+matrix('MATRIX',[[3.]]))
            elif fault=='allocation_small' and kind=='FIT_RHS_BLOCK':events[i]=(kind,body.replace('10 6 ','10 1 '))
    write_events(tmp_path,events,version=3 if fault=='raw_in_schema3' else 4)
    with pytest.raises(ValueError):reader.replay(tmp_path,expected_frequencies=2)

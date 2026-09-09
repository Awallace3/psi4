"""Schema6 subset/density provenance; not full integral-cache provenance."""
import copy
import importlib.util
from pathlib import Path
import numpy as np
import pytest

spec=importlib.util.spec_from_file_location('subset_density_reader',Path(__file__).parent/'data_isapol/oracle/replay_response.py')
r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)


def fixture():
    events=[]
    def add(kind,**kw):
        e=dict(version=6,header=(len(events)+1,1,4,2,2,2,4),kind=kind,**kw)
        events.append(e);return e
    for kind,count in [('OO',3),('OV',4),('VV',3)]:
        identity=(kind,kind.lower())
        for row in range(1,count+1):
            add(kind+'_ROW',destination=identity,parent=('NN','nn'),ar=row,
                ij=r.subset_parent_index(kind,row,2,2),row=np.array([row,-row],float))
        add('FIT_METADATA',identity=identity,shape=(count,2),fit=((1,0,3,1),(0.,0.,0.,0.),(0,0)))
    d=add('KERNEL_DENSITY',density_identity=('OO','oo'),coefficients=np.array([8.,-8.]))
    add('KERNEL_SOURCE',identity=('K','k'),density_identity=('OO','oo'),density_serial=d['header'][0],constrained=0)
    return events


def test_subset_density_generation_and_ordered_sum():
    result=r.validate_subset_events(fixture())
    assert [g['kind'] for g in result['generations']]==['OO','OV','VV']
    assert result['density_reconstruction'][0]['max_absolute']==0
    assert result['complete_parent_solve_and_cache_provenance'] is False


@pytest.mark.parametrize('kind,expected',[('OO',[1,2,3]),('OV',[4,5,7,8]),('VV',[6,9,10])])
def test_subset_parent_indices(kind,expected):
    assert [r.subset_parent_index(kind,i,2,2) for i in range(1,len(expected)+1)]==expected


@pytest.mark.parametrize('damage',['wrong_vv','duplicate','missing_setter','wrong_density','unknown_density','changed_metadata'])
def test_subset_density_rejects_broken_association(damage):
    events=fixture()
    if damage=='wrong_vv':next(e for e in events if e['kind']=='VV_ROW')['ij']=1
    elif damage=='duplicate':next(e for e in events if e['kind']=='VV_ROW' and e['ar']==2)['ar']=1
    elif damage=='missing_setter':events=[e for e in events if not(e['kind']=='FIT_METADATA' and e['identity'][0]=='VV')]
    elif damage=='wrong_density':events[-2]['density_identity']=('VV','vv')
    elif damage=='unknown_density':events[-1]['density_serial']=1
    else:
        e=copy.deepcopy(next(e for e in events if e['kind']=='FIT_METADATA'))
        e['fit']=((1,0,3,1),(1.,0.,0.,0.),(0,0));events.append(e)
    with pytest.raises(ValueError):r.validate_subset_events(events)


def test_density_disagreement_is_a_measured_gate_not_repaired():
    events=fixture();events[-2]['coefficients'][0]+=1
    assert r.validate_subset_events(events)['density_reconstruction'][0]['max_scaled']>1e-9


def test_density_keeps_original_subset_after_later_overwrite():
    events=fixture();original_source=r.validate_subset_events(events)['density_reconstruction'][0]['source_start']
    replacements=copy.deepcopy(events[:4])
    for e in replacements:
        e['header']=(len(events)+1,)+e['header'][1:]
        if e['kind']=='OO_ROW':e['row']*=10
        events.append(e)
    source=copy.deepcopy(events[14]);source['header']=(len(events)+1,)+source['header'][1:]
    assert source['kind']=='KERNEL_SOURCE'
    events.append(source)
    result=r.validate_subset_events(events)
    assert result['density_reconstruction'][0]['source_start']==original_source
    assert result['density_reconstruction'][0]['max_absolute']==0


def test_cached_nn_to_ov_retag_preserves_original_generation():
    events=fixture();meta=next(e for e in events if e['kind']=='FIT_METADATA' and e['identity'][0]=='OV')
    meta['fit']=((1,0,1,1),(0.,0.,0.,0.),(0,0))
    refresh=copy.deepcopy(meta);refresh['header']=(len(events)+1,)+refresh['header'][1:]
    refresh['fit']=((1,0,3,1),(0.,0.,0.,0.),(0,0));events.append(refresh)
    result=r.validate_subset_events(events)
    assert len(result['generations'])==3 and len(result['metadata_refreshes'])==1
    assert result['generations'][1]['fit'][0][2]==1
    assert result['metadata_refreshes'][0]['after'][0][2]==3
    bad=copy.deepcopy(events);bad[-1]['fit']=((1,0,3,1),(1.,0.,0.,0.),(0,0))
    with pytest.raises(ValueError,match='Changed subset metadata'):r.validate_subset_events(bad)


def parent_fixture():
    events=fixture()
    fit=((1,0,1,1),(0.,0.,0.,0.),(0,0))
    for e in events:
        e['version']=7
        if e['kind']=='FIT_METADATA':e['fit']=fit
    begin=dict(version=7,kind='SOLVE_BEGIN',header=(1,1,4,2,2,2,4),
               fit_control=(1,4,10,1,0,0),x_identity=('NN','nn'),fit=fit)
    end=dict(version=7,kind='SOLVE_END',header=(2,1,4,2,2,2,4),solve_end=(1,0))
    for e in events:
        e['header']=(e['header'][0]+2,)+e['header'][1:]
        if e['kind']=='KERNEL_SOURCE':e['density_serial']+=2
    return [begin,end]+events


def test_successful_parent_links_all_subset_generations():
    result=r.validate_subset_events(parent_fixture())
    assert result['complete_parent_solve_provenance'] is True
    assert result['complete_parent_solve_and_cache_provenance'] is False
    assert {g['parent_solve_id'] for g in result['generations']}=={1}


@pytest.mark.parametrize('damage',['missing_end','failed_end','wrong_parent','wrong_fit','nested','unfinished'])
def test_parent_generation_rejects_incomplete_or_mismatched_origin(damage):
    events=parent_fixture()
    if damage=='missing_end':events.pop(1)
    elif damage=='failed_end':events[1]['solve_end']=(1,-1)
    elif damage=='wrong_parent':events[0]['x_identity']=('other','other')
    elif damage=='wrong_fit':events[0]['fit']=((1,0,1,1),(1.,0.,0.,0.),(0,0))
    elif damage=='nested':events.insert(1,copy.deepcopy(events[0]))
    else:
        b=copy.deepcopy(events[0]);b['fit_control']=(2,4,10,1,0,0);events.append(b)
    with pytest.raises(ValueError):r.validate_subset_events(events)


def test_schema7_parent_begin_parser(tmp_path):
    text=('ISAPOL_RESPONSE_EVENT 7\n1 1 4 2 2 2 4\nSOLVE_BEGIN\nM\ndalton\n'
          '1 4 10 1 0 0\nS___\n_A_A\nT_MO\n___A\nA\na\nB\nb\nNN\nnn\n'
          '2 2 10 2 10 2\n1 0 1 1\nFIT_PARAMETERS\n4\n0 0 0 0\n0 0\nEND_RESPONSE_EVENT\n')
    path=tmp_path/'solve.dat';path.write_text(text)
    assert r.read_event(path)['fit'][1][0]==0.
    path.write_text(text.replace('S___','Sc__'))
    with pytest.raises(ValueError,match='metric/constraint'):r.read_event(path)


def test_schema6_vv_reader_checks_full_nn_offset(tmp_path):
    text='ISAPOL_RESPONSE_EVENT 6\n1 1 4 2 2 2 4\nVV_ROW\nM\ndalton\nVV\nvv\nNN\nnn\n1 6\nROW\n2\n1 -1\nEND_RESPONSE_EVENT\n'
    path=tmp_path/'row.dat';path.write_text(text)
    assert r.read_event(path)['ij']==6
    path.write_text(text.replace('1 6\nROW','1 1\nROW'))
    with pytest.raises(ValueError,match='Packed'):r.read_event(path)

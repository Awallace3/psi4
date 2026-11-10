"""Gauge equivalence must never become arbitrary rotations of energy entries."""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from repeat_gauge import compare_arrays as compare,compare_repeat,FULL,PAIRS,WATER_DIMENSIONS
from gpu_runtime import digest


def dimensions(occupied=(3,3),ao=3,atoms=2,frozen=(1,1)):
    return {"atoms":atoms,"ao":ao,"occupied":list(occupied),"frozen":list(frozen),
            "active":[n-nf for n,nf in zip(occupied,frozen)]}


def check(*args,case=None):
    return compare(*args,dimensions() if case is None else case)


def fixture(mapping=(0,2,1),angle=.31):
    rng=np.random.default_rng(73)
    l={"ZA":np.array([1.,0.]),"ZB":np.array([0.,1.])}
    r={key:value.copy() for key,value in l.items()}
    count=len(mapping)
    lh={"S":np.eye(count)};rh={"S":np.eye(count)}
    indices=np.r_[np.arange(2),2+np.asarray(mapping),2+count]
    signs=np.ones(count);signs[1]=-1
    k=np.eye(count);k[-2:,-2:]=[[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]]
    for side in "AB":
        l["Locc_"+side]=np.eye(count);l["Uocc0"+side]=np.eye(count)
        q=np.linspace(.2,.8,count);q[0]=1.
        l["Qocc0"+side]=np.array([q,1-q])
        r["Locc_"+side]=l["Locc_"+side][:,mapping]*signs
        r["Uocc0"+side]=k.T@r["Locc_"+side]
        r["Qocc0"+side]=l["Qocc0"+side][:,mapping]
        lh["Cocc_"+side]=np.eye(count);rh["Cocc_"+side]=k.copy()
        lh["eps_occ_"+side]=np.r_[-20.,-np.ones(count-1)]
        rh["eps_occ_"+side]=lh["eps_occ_"+side].copy()
    for name in FULL:
        l[name]=rng.normal(size=(count+3,count+3))
        r[name]=l[name][np.ix_(indices,indices)].copy()
    for name in PAIRS:
        l[name]=rng.normal(size=(count-1,count-1))
        active=np.asarray(mapping)[1:]-1
        r[name]=l[name][np.ix_(active,active)].copy()
    return l,r,lh,rh


class RepeatGauge(unittest.TestCase):
    def test_unequal_spaces_and_directional_fields_keep_AB_axes(self):
        rng=np.random.default_rng(81);natom=2;nbf=5
        l={"ZA":np.array([1.,0.]),"ZB":np.array([0.,1.])}
        r={k:v.copy() for k,v in l.items()}
        lh={"S":np.eye(nbf)};rh={"S":np.eye(nbf)}
        maps=[np.array([0,2,1]),np.array([0,1,3,2])]
        for side,mapping in zip("AB",maps):
            count=len(mapping);co=np.eye(nbf)[:,:count]
            signs=np.ones(count);signs[-1]=-1
            l["Locc_"+side]=co;l["Uocc0"+side]=np.eye(count)
            q=rng.random((natom,count));q/=q.sum(axis=0)
            l["Qocc0"+side]=q
            r["Locc_"+side]=co[:,mapping]*signs
            r["Uocc0"+side]=np.eye(count)[:,mapping]*signs
            r["Qocc0"+side]=q[:,mapping]
            lh["Cocc_"+side]=co;rh["Cocc_"+side]=co.copy()
            lh["eps_occ_"+side]=np.r_[-20.,-np.ones(count-1)]
            rh["eps_occ_"+side]=lh["eps_occ_"+side].copy()
        indices=[np.r_[np.arange(natom),natom+m,natom+len(m)] for m in maps]
        for name in FULL:
            l[name]=rng.normal(size=(6,7))
            r[name]=l[name][np.ix_(*indices)].copy()
        for name in PAIRS:
            l[name]=rng.normal(size=(2,3))
            r[name]=l[name][np.ix_(maps[0][1:]-1,maps[1][1:]-1)].copy()
        report=check(l,r,lh,rh,case=dimensions((3,4),ao=5))
        self.assertLess(max(report["maximum_absolute_field_errors"].values()),1e-14)

    def test_only_orbital_permutation_signs_and_proven_canonical_frame_are_allowed(self):
        result=check(*fixture())
        self.assertEqual(result["left_indices_for_right_occupied"],[[0,2,1],[0,2,1]])
        self.assertEqual(result["frozen_occupied"],[1,1])
        self.assertLess(max(result["maximum_absolute_field_errors"].values()),1e-14)

    def test_empty_frozen_blocks_are_valid_without_dropping_checks(self):
        l,r,lh,rh=fixture()
        m=np.array([0,2,1])
        for name in PAIRS:
            l[name]=np.arange(9,dtype=float).reshape(3,3)
            r[name]=l[name][np.ix_(m,m)].copy()
        with self.assertRaises(ValueError):check(l,r,lh,rh)
        self.assertEqual(check(l,r,lh,rh,case=dimensions(frozen=(0,0)))["frozen_occupied"],[0,0])

    def test_unchanged_repeat_tolerance_still_rejects_real_field_errors(self):
        for name in (*FULL,*PAIRS,"Qocc0B","Uocc0A","Locc_A","ZA"):
            l,r,lh,rh=fixture()
            r[name].flat[-1]+=1e-5
            with self.subTest(name=name),self.assertRaises((AssertionError,RuntimeError,ValueError)):
                check(l,r,lh,rh)

    def test_frozen_active_swaps_and_continuous_localized_rotations_fail(self):
        l,r,lh,rh=fixture()
        for side in "AB":
            angle=.1
            rotation=np.eye(3)
            rotation[1:,1:]=[[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]]
            r["Locc_"+side]=r["Locc_"+side]@rotation
            r["Uocc0"+side]=r["Uocc0"+side]@rotation
        with self.assertRaises(AssertionError):check(l,r,lh,rh)
        l,r,lh,rh=fixture(mapping=(1,0,2))
        with self.assertRaises((AssertionError,RuntimeError)):check(l,r,lh,rh)

    def test_energy_matrices_are_permuted_not_rotated_or_orbital_sign_scaled(self):
        l,r,lh,rh=fixture()
        r["Elst_AB"][3,:]*=-1
        with self.assertRaises(AssertionError):check(l,r,lh,rh)
        l,r,lh,rh=fixture()
        rotation=np.eye(6);rotation[3:5,3:5]=[[.8,-.6],[.6,.8]]
        r["Exch_AB"]=rotation.T@r["Exch_AB"]@rotation
        with self.assertRaises(AssertionError):check(l,r,lh,rh)

    def test_non_degenerate_canonical_energy_mixing_cannot_hide_in_U(self):
        l,r,lh,rh=fixture()
        lh["eps_occ_A"]=np.array([-20.,-1.,-.5])
        rh["eps_occ_A"]=lh["eps_occ_A"].copy()
        with self.assertRaisesRegex(AssertionError,"energy/frame"):check(l,r,lh,rh)

    def test_missing_or_nonfinite_fields_cannot_be_dropped(self):
        l,r,lh,rh=fixture();del r["Ind20r_BA"]
        with self.assertRaises(ValueError):check(l,r,lh,rh)
        l,r,lh,rh=fixture();r["Disp_AB"][1,2]=np.nan
        with self.assertRaises(ValueError):check(l,r,lh,rh)

    def test_non_self_inverse_cycle_keeps_mapping_direction(self):
        result=check(*fixture(mapping=(0,2,3,1)),case=dimensions((4,4),ao=4))
        self.assertEqual(result["left_indices_for_right_occupied"],[[0,2,3,1]]*2)

    def test_rectangular_coefficients_in_nonidentity_AO_metric(self):
        l,r,lh,rh=fixture()
        metric=np.diag(np.arange(1.,6.))
        co=np.diag(1/np.sqrt(np.arange(1.,6.)))[:,:3]
        for side in "AB":
            k=rh["Cocc_"+side]
            l["Locc_"+side]=co@l["Uocc0"+side]
            r["Locc_"+side]=co@k@r["Uocc0"+side]
            lh["Cocc_"+side]=co;rh["Cocc_"+side]=co@k
        lh["S"]=metric;rh["S"]=metric.copy()
        check(l,r,lh,rh,case=dimensions(ao=5))


def write_water_fixture(root):
    """Real wrapper boundary: independent paths/PIDs and hashed raw artifacts."""
    root=Path(root);rng=np.random.default_rng(91)
    left={"ZA":np.array([8.,1.,1.,0.,0.,0.]),"ZB":np.array([0.,0.,0.,8.,1.,1.])}
    right={k:v.copy() for k,v in left.items()}
    lh={"S":np.eye(82)};rh={"S":np.eye(82)}
    mappings=[np.arange(5),np.array([0,2,1,3,4])]
    for side,mapping in zip("AB",mappings):
        co=np.eye(82)[:,:5]
        left["Locc_"+side]=co;right["Locc_"+side]=co[:,mapping]
        left["Uocc0"+side]=np.eye(5);right["Uocc0"+side]=np.eye(5)[:,mapping]
        q=rng.random((6,5));q/=q.sum(axis=0)
        left["Qocc0"+side]=q;right["Qocc0"+side]=q[:,mapping]
        for raw in (lh,rh):
            raw["Cocc_"+side]=co.copy()
            raw["eps_occ_"+side]=np.array([-20.,-1.,-.8,-.6,-.5])
    indices=[np.r_[np.arange(6),6+m,11] for m in mappings]
    for name in FULL:
        left[name]=rng.normal(size=(12,12))
        right[name]=left[name][np.ix_(*indices)].copy()
    for name in PAIRS:
        left[name]=rng.normal(size=(4,4))
        right[name]=left[name][np.ix_(mappings[0][1:]-1,mappings[1][1:]-1)].copy()
    paths=[]
    for label,arrays,raw,pid in (("baseline",left,lh,101),("repeat",right,rh,102)):
        path=root/label;path.mkdir();paths.append(path)
        np.savez(path/"fragment-matrices.npz",**arrays);np.savez(path/"raw-hf.npz",**raw)
        doc={"status":"EXECUTED_REQUIRES_INDEPENDENT_COMPARISON","arm":"gpu","pid":pid,
            "source_pin":"science","runtime_config_sha256":"config","geometry_sha256":"geometry",
            "scaled":True,"orbital_dimensions":WATER_DIMENSIONS,
            "options":{"basis":"aug-cc-pvdz","fisapt_fsapt_filepath":str(path/"fsapt")},
            "artifact_sha256":{n:digest(path/n) for n in ("raw-hf.npz","fragment-matrices.npz")},
            "energy_Eh":-.008,"Disp20,u":-.002,"Exch-Disp20,u":.001}
        (path/"result.json").write_text(json.dumps(doc))
    return paths


class RepeatWrapper(unittest.TestCase):
    def test_only_the_per_child_output_option_is_normalized(self):
        with tempfile.TemporaryDirectory() as directory:
            paths=write_water_fixture(directory);compare_repeat(*paths)
            path=paths[1]/"result.json";doc=json.loads(path.read_text())
            doc["options"]["basis"]="different";path.write_text(json.dumps(doc))
            with self.assertRaisesRegex(RuntimeError,"scientific options"):compare_repeat(*paths)

    def test_self_comparison_and_same_PID_are_not_fresh_repeats(self):
        with tempfile.TemporaryDirectory() as directory:
            paths=write_water_fixture(directory)
            with self.assertRaisesRegex(RuntimeError,"Distinct fresh-process"):
                compare_repeat(paths[0],paths[0])
            path=paths[1]/"result.json";doc=json.loads(path.read_text());doc["pid"]=101
            path.write_text(json.dumps(doc))
            with self.assertRaisesRegex(RuntimeError,"process identities"):compare_repeat(*paths)

    def test_metadata_cannot_redefine_fixed_frozen_case(self):
        with tempfile.TemporaryDirectory() as directory:
            paths=write_water_fixture(directory)
            for path in paths:
                p=path/"result.json";doc=json.loads(p.read_text())
                doc["orbital_dimensions"]["frozen"]=[0,0];doc["orbital_dimensions"]["active"]=[5,5]
                p.write_text(json.dumps(doc))
            with self.assertRaisesRegex(RuntimeError,"fixed frozen water"):compare_repeat(*paths)

    def test_changed_artifact_or_wrong_output_origin_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            paths=write_water_fixture(directory)
            with (paths[1]/"raw-hf.npz").open("ab") as stream:stream.write(b"changed")
            with self.assertRaisesRegex(RuntimeError,"artifact changed"):compare_repeat(*paths)
        with tempfile.TemporaryDirectory() as directory:
            paths=write_water_fixture(directory);path=paths[1]/"result.json"
            doc=json.loads(path.read_text());doc["options"]["fisapt_fsapt_filepath"]=str(paths[0]/"fsapt")
            path.write_text(json.dumps(doc))
            with self.assertRaisesRegex(RuntimeError,"own fresh directory"):compare_repeat(*paths)


if __name__=="__main__":unittest.main()

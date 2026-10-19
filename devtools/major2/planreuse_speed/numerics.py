"""Host-only snapshot analysis/storage, NEVER called inside timing windows."""
from pathlib import Path
import numpy as np
import config as s
import fixed_jk as f
import observe as o


class Store:
    def __init__(self, root, cap=s.PROCESS_CAP):
        self.root, self.cap, self.used, self.saved = Path(root), cap, 0, []

    def save(self, label, arrays):
        s.require(label in ("warmup", "round5-control") and label not in self.saved and
                  set(arrays) == {"J", "K"}, "only warmup and final control J/K stored")
        s.require(self.used+sum(v.nbytes+128 for v in arrays.values()) <= self.cap,
                  "storage pre-reservation cap")
        result = {key:f.save_array(self.root/(label+"-"+key+".npy"), v)
                  for key, v in arrays.items()}
        self.used += sum((self.root/v["file"]).stat().st_size for v in result.values())
        s.require(self.used <= self.cap, "storage actual cap")
        for key, meta in result.items():
            path = self.root/meta["file"]
            s.require(not path.stat().st_mode & 0o222 and
                      o.array_info(np.load(path, allow_pickle=False)) == meta["array"],
                      "readonly snapshot readback")
        self.saved.append(label)
        return result


class Analysis:
    def __init__(self, store, shape=(1786, 1786)):
        self.store, self.shape = store, shape
        self.reference = None
        self.reference_d = None
        self.scalar = None

    def snapshot(self, phase, round_id, arrays, density):
        for v in [*arrays.values(), density]:
            f.finite_array(v, self.shape)
        s.require(set(arrays) == {"J", "K"}, "exact J/K set")
        info = {k:o.array_info(v) for k, v in arrays.items()}
        d_info = o.array_info(density)
        scalar = f.scalar(density, arrays["J"], arrays["K"])
        if self.reference is None:
            s.require(phase == "warmup" and round_id == 0, "first snapshot warmup")
            self.reference = arrays
            self.reference_d = density
            self.scalar = scalar
        differences = {k:f.difference(self.reference[k], v) for k, v in arrays.items()}
        d_difference = f.difference(self.reference_d, density)
        scalar_delta = scalar-self.scalar
        s.require(np.isfinite(scalar_delta), "nonfinite scalar delta")
        symmetric = {k:float(np.max(np.abs(v-v.T))) for k, v in arrays.items()}
        scalar_breach = abs(scalar_delta) > f.BOUNDS["scalar_abs_eh"]
        findings = dict(matrices=differences, scalar_delta_eh=scalar_delta,
                        scalar_exceeds_bound=scalar_breach,
                        symmetry_maxabs=symmetric,
                        symmetry_findings={k:v > f.BOUNDS["symmetry_maxabs"]
                                           for k, v in symmetric.items()})
        stable = not scalar_breach and not any(
            v["exceeds_maxabs"] or v["exceeds_relative_frobenius"]
            for v in differences.values())
        saved = {}
        if phase == "warmup" or (phase == "control" and round_id == 5):
            saved = self.store.save("warmup" if phase == "warmup" else "round5-control", arrays)
        return dict(arrays=info, actual_D=d_info, actual_D_difference=d_difference,
                    matched_D=d_info == o.array_info(self.reference_d),
                    scalar_eh=scalar, findings=findings,
                    stable_within_original_roundoff=stable, saved=saved)


def total_storage(raw):
    files = list(Path(raw).glob("speed-*/*.npy"))
    s.require(len(files) <= 24 and sum(p.stat().st_size for p in files) <= s.CAP,
              "global 24 array/2GiB cap")
    return sum(p.stat().st_size for p in files)

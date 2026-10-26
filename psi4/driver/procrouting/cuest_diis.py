"""Device SCF DIIS bulk operations with Psi4's existing small host solvers.

The NVIDIA cuEST example motivates resident history and incremental Gram
updates. We intentionally retain conditioned least-squares and A/EDIIS
blending instead of copying its illustrative unconditioned matrix inverse.
No full-history host matrices or ADIIS difference matrices are retained.
"""
import numpy as np

from psi4 import core
from ..p4util.exceptions import ValidationError
from .diis import DIIS, RemovalPolicy, StoragePolicy


def validate_cuest_diis(wfn):
    """Validate explicitly requested device mode before allocating JK/DIIS."""
    if not core.get_global_option("CUEST_DIIS"):
        return
    if not core.get_global_option("USE_CUEST"):
        raise ValidationError("CUEST_DIIS requires USE_CUEST true.")
    if not hasattr(core, "CuestDIIS"):
        raise ValidationError("CUEST_DIIS requires a Psi4 build with cuEST and CUDA.")
    if core.get_option("SCF", "REFERENCE") not in {"RHF", "RKS", "UHF", "UKS"} or wfn.nirrep() != 1:
        raise ValidationError("CUEST_DIIS supports only C1 RHF/RKS and UHF/UKS references.")
    if core.get_option("SCF", "ORBITAL_OPTIMIZER_PACKAGE") != "INTERNAL":
        raise ValidationError("CUEST_DIIS requires the INTERNAL orbital optimizer.")
    if not 1 <= core.get_option("SCF", "DIIS_MAX_VECS") <= 128:
        raise ValidationError("CUEST_DIIS requires DIIS_MAX_VECS between 1 and 128.")
    if core.get_global_option("CUEST_DIIS_MAX_MEMORY_MB") <= 0:
        raise ValidationError("CUEST_DIIS_MAX_MEMORY_MB must be positive.")
    if not core.get_option("SCF", "DIIS") and core.get_option("SCF", "SCF_INITIAL_ACCELERATOR") == "NONE":
        raise ValidationError("CUEST_DIIS requires DIIS or an initial A/EDIIS accelerator.")


class CuestDIIS(DIIS):
    is_cuest_diis = True

    def __init__(self, max_vecs, name, *, closed_shell=True, engines=None, stats=None):
        engines = {"diis"} if engines is None else set(engines)
        if not hasattr(core, "CuestDIIS"):
            raise ValidationError("CUEST_DIIS requires a Psi4 build with cuEST and CUDA.")
        if not engines or not engines <= {"diis", "adiis", "ediis"} or {"adiis", "ediis"} <= engines:
            raise ValidationError(f"Unsupported cuEST DIIS engine combination: {engines}")
        super().__init__(max_vecs, name, RemovalPolicy.LargestError, StoragePolicy.InCore,
                         closed_shell, engines)
        self.backend = core.CuestDIIS(
            max_vecs, core.get_global_option("CUEST_DIIS_MAX_MEMORY_MB"),
            "diis" in engines, bool(engines & {"adiis", "ediis"}),
            self.removal_policy == RemovalPolicy.OldestAdded)
        self.stats_record = {} if stats is None else stats
        self._closed = False

    def _publish(self):
        self.stats_record.update(self.backend.stats())
        self.stats_record["stored_vectors"] = self.backend.size()

    def _require_open(self):
        if self._closed:
            raise ValidationError("cuEST DIIS history has been deleted; construct a new manager.")

    def _record_entry(self, nspin, energy=None):
        # Only scalar energy bookkeeping lives on host. Targets, errors and
        # densities stay in native device storage, including for A/EDIIS.
        entry = {}
        if "ediis" in self.engines:
            if energy is None or not np.isfinite(energy):
                raise ValidationError("cuEST EDIIS requires a finite energy for each entry.")
            entry["energy"] = [float(energy)]
        pivot = self.backend.last_pivot()
        if pivot == len(self.stored_vectors):
            self.stored_vectors.append(entry)
        else:
            self.stored_vectors[pivot] = entry
        self.template = {"target": [None] * nspin}
        if self.engines & {"adiis", "ediis"}:
            self.template["densities"] = [None] * nspin
        self.iter_num += 1
        self.max_vecs = self.backend.capacity()
        self._publish()

    def scf_update(self, focks, densities, overlap, orthogonalizer, *, save, rms, energy=None):
        self._require_open()
        if save and "ediis" in self.engines and (energy is None or not np.isfinite(energy)):
            raise ValidationError("cuEST EDIIS requires a finite energy for each entry.")
        metric = self.backend.scf_update(focks, densities, overlap, orthogonalizer, save, rms)
        if save:
            self._record_entry(len(focks), energy)
        self._publish()
        return metric

    def add_entry(self, entry):
        """Matrix-entry seam for independent host/device DIIS parity tests."""
        self._require_open()
        if "ediis" in self.engines and (
                len(entry.get("energy", [])) != 1 or not np.isfinite(entry["energy"][0])):
            raise ValidationError("cuEST EDIIS requires a finite energy for each entry.")
        self.backend.add_entry(entry.get("error", []), entry["target"], entry.get("densities", []))
        self._record_entry(len(entry["target"]), entry.get("energy", [None])[0])
        return True

    def get_dot_product(self, i, j):
        n = self.backend.size()
        return np.asarray(self.backend.error_gram()).reshape(n, n)[i, j]

    def _density_fock_gram(self):
        n = self.backend.size()
        return np.asarray(self.backend.density_fock_gram()).reshape(n, n)

    def adiis_populate(self):
        q = self._density_fock_gram()
        # Same latest-array-slot convention as the existing host implementation,
        # including oldest-added wraparound. No O(history * N^2) difference copies.
        self.adiis_linear = q[:, -1] - q[-1, -1]
        self.adiis_quadratic = q - q[:, -1, None] - q[None, -1, :] + q[-1, -1]
        if self.closed_shell:
            self.adiis_linear *= 2
            self.adiis_quadratic *= 2

    def ediis_populate(self):
        q = self._density_fock_gram()
        diagonal = np.diag(q)
        self.ediis_quadratic = -0.5 * (diagonal[:, None] + diagonal - q - q.T)
        if self.closed_shell:
            self.ediis_quadratic *= 2

    def extrapolate(self, *args, Dnorm=None):
        self._require_open()
        coefficients, performed = self.extrapolation_coefficients(Dnorm)
        if coefficients is not None:
            self.backend.extrapolate(list(coefficients), list(args))
        self._publish()
        return performed

    def reset_subspace(self):
        super().reset_subspace()
        if hasattr(self, "backend"):
            self.backend.reset_subspace()
            self._publish()

    def delete_diis_file(self):
        # Do not touch PSIF_LIBDIIS: this manager never acquires that resource.
        if not self._closed:
            self._publish()
            self.stats_record["history_vectors_before_cleanup"] = self.backend.size()
            self.backend.clear()
            self._publish()
            self.stats_record["device_bytes_retained_after_cleanup"] = 0
            self._closed = True

"""Observe existing native-plan lifetimes; never change their math or order."""
from contextlib import contextmanager
import subprocess
import time


def observed_factory(runtime):
    observations=runtime.report.setdefault("full_device_observations",[])
    uuid=runtime.report["primitive_probe"]["device"]["uuid"]

    def snapshot(label,provider=None,owner=None):
        if provider is not None:
            provider.session.synchronize_device()
        raw=subprocess.check_output(["nvidia-smi","--id="+uuid,
            "--query-gpu=uuid,memory.total,memory.free,memory.used",
            "--format=csv,noheader,nounits"],text=True,timeout=20)
        values=[s.strip() for s in raw.strip().split(",")]
        if len(values)!=4 or values[0]!=uuid:
            raise RuntimeError("Full-run GPU memory observation changed device")
        row={"label":label,"monotonic_seconds":time.monotonic(),"uuid":uuid,
             "nvml_total_MiB":int(values[1]),"nvml_free_MiB":int(values[2]),
             "nvml_used_MiB":int(values[3]),"sampled_not_peak":True}
        if provider is not None:
            session=provider.session
            free,total=session.rt(session.runtime.cudaMemGetInfo())
            row.update(cuda_free_bytes=int(free),cuda_total_bytes=int(total),
                tracked_live_bytes=session.live_bytes,
                declared_opaque_reserve_bytes=session.native_bytes,
                native_plan=(owner if owner is not None else provider.owned_plan()).report())
        observations.append(row)

    @contextmanager
    def factory(primary,auxiliary):
        # Includes the still-live upstream cuEST JK allocation. No claim that
        # the session's opaque reservation measures those native internals.
        snapshot("before-fragment-factory-upstream-JK-still-live")
        with runtime.provider(primary,auxiliary) as provider:
            owner=provider.owned_plan()
            snapshot("JKFIT-plan-constructed",provider,owner)
            original=provider.sibling_provider
            @contextmanager
            def sibling(aux):
                with original(aux) as ri:
                    snapshot("RIFIT-and-JKFIT-plans-coexist-before-terms",ri)
                    yield ri
                    snapshot("RIFIT-and-JKFIT-before-RIFIT-cleanup",ri)
            provider.sibling_provider=sibling
            try:
                yield provider
                snapshot("JKFIT-before-final-cleanup",provider,owner)
            finally:
                provider.sibling_provider=original
        snapshot("fragment-factory-cleaned-up-upstream-JK-may-still-live")
    return factory

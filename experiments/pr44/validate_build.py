import hashlib
import json
from pathlib import Path
import sys
import psi4

build = Path(sys.argv[1]).resolve()
core = Path(psi4.core.__file__).resolve()
assert core.is_relative_to(build / "stage")
digest = hashlib.sha256()
with core.open("rb") as handle:
    for block in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(block)
data = {
    "commit": sys.argv[2], "core": str(core),
    "core_sha256": digest.hexdigest(),
    "version": psi4.__version__,
}
(build / "validated.json").write_text(json.dumps(data, indent=2) + "\n")
print(json.dumps(data), flush=True)

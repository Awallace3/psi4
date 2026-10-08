# ISA and numerical reference tools

Tracked tests import these modules with portable inputs. They do not require
CamCASP, ORIENT or a reference executable.

- `reconstruct_isa_basis.py`: independent polynomial reconstruction of exported
  CamCASP basis descriptors; `../../test_isapol_basis.py` uses it as an oracle.

# ISA and numerical reference tools

Tracked tests import these modules with portable inputs. They do not require
CamCASP, ORIENT or a reference executable.

- `reconstruct_isa_basis.py`: independent polynomial reconstruction of exported
  CamCASP basis descriptors; `../../test_isapol_basis.py` uses it as an oracle.
- `extract_lw_hermetic.py`: strict parser for the static historical dialect,
  plus an opt-in CLI that reads only an explicitly supplied source directory.
- `extract_lw_dynamic.py`: strict parser and printed-frequency check for the
  dynamic dialect. Its original capture CLI depended on private development
  paths and an unshipped hash inventory, so it was removed; the parser is
  unchanged.
- `../../test_isapol_lw_{hermetic,dynamic}.py` test these parsers directly on
  synthetic documents. The tests never run a CLI.

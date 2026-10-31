# Opt-in sealed runtime integration — not yet GPU-qualified

Target: the qualified water profile still makes163,164 complete17-file lease
checks per energy call. The latest accepted small-water result remains42–44s.
The separately captured loader prototype passed CPU and bounded GPU compatibility
checks; see [the measured ledger](HOT_PATH_ADMISSION.md).

## Contract

`process-owned-kernel-sealed-snapshots-v1` is a separate adapter, not a weakened
lease checker. Fresh-byte and kernel-read-lease selections remain available.

- A normal import of the coherently installed, stdlib-only
  `psi4_runtime_support` package prepares snapshots **before** Psi4/core/vendor
  imports. No scientific module is injected into `sys.modules` or overlaid.
- Preparation copies opened source descriptors, checks source stability/SHA256,
  installs irreversible WRITE/GROW/SHRINK/SEAL/EXEC seals, then hashes the sealed
  destination again. The complete snapshot allocation is bounded to2GiB.
- The parent creates a fresh owned0700 diagnostic directory, **excluded** from
  the loader path along with original SDK/toolkit directories. First loads use
  only `/proc/self/fd/N` for the verified snapshots, never mutable aliases.
  Bounded ELF parsing obtains DT_NEEDED/SONAME from sealed bytes. Pinned
  dependencies are loaded recursively and must bind their exact SONAME through
  `RTLD_NOLOAD`; unknown non-base dependencies and non-ORIGIN search paths reject.
  The small readonly CPU/OS base remains outside the toolkit snapshot contract.
- Snapshot descriptors and loaded handles remain owned until process exit,
  independently of session/plan cleanup. There is no close, transfer, renewal,
  fresh-hash fallback or contract switching.
- Once loaded, the backing inode is immutable even if its original file, alias
  or descriptor is removed. `check_live()` therefore checks caller/lifetime
  state in constant time, not filesystem metadata or revocable SIGIO leases.
  Unloaded snapshots still require descriptor/seal validation before loading.
- The scientific selector adopts only the exact prepared object retained by
  the support package. Both preparation and selection share the admission lock;
  legacy-session selection during preparation/handoff is rejected.
- Published proof failures terminate the dedicated worker without native
  unwinding. Ordinary pre-publication input/copy/seal failures roll back.

## Mapping obligations are retained

Every existing factory/provider mapping audit still runs. The sealed parser
translates only exact admitted kernel-name/inode/device triples to logical
source locators; no stale parsed rows survive a call. Foreign/unsealed toolkit
images, including cuDNN/CUPTI, and unrelated deleted shared libraries reject.
The legacy parser's deleted-library rejection is unchanged.

Bootstrap checks all25 canonical native/factory symbol owners before core import
and again after it. Later native context/version/FP64, device, buffer, numerical
and cleanup checks remain in place. No loader epoch or cached audit substitutes
for the retained fresh scans.

The whole-build proof includes the complete support-package inventory in
addition to scientific Python/core/dependency inventories. Parent-created
per-worker aliases and cache homes are separate from durable SDK/build inputs.
Sealed mode needs no reusable tmpfs dependency copy to work around Btrfs mapping
identities: its executable inodes are the newly verified memfds.

## Validation and remaining gates

CPU fixtures exercise real kernel seals and tiny ELF images, source changes,
bounded copying, rollback, foreign mappings, normal early support import/adoption,
contract/thread/fork misuse and absence of file IO on loaded-image hot checks.
These fixtures are not CUDA/cuEST/device evidence.

The complete host run passed373+33+109=**515 tests**, without skips.
Review reproduced a first-load alias defect in the initial candidate: replacing
a dependency alias executed a foreign constructor before any mapping rejection.
That regression was run red, then green after descriptor-bound loading; primary
alias substitution is also covered. The alias-based candidate is not accepted.
The separate descriptor-only vendor CPU probe passed on Cachy with all nine
sealed preloads and NVRTC compilation, without a GPU call:
`sealed-descriptor-cpu-01/result.json`, SHA256
`d30c4a723fe0aa81ef44cbdb5283c6d741e7b188c9c7d7b6646125dc1a7496a8`.
Code is captured on `prototype/cuest-sealed-runtime` at`8fd153b61b`.
The bounded descriptor-only GPU repeat also passed one FP64 kernel (`2 -> 3`),
cuEST0.2.50 FP64 context construction and every explicit cleanup step:
`sealed-descriptor-gpu-01/result.json`, SHA256
`0e19acc6cc5cba9cfaa5ced72283a91b9a2048bfc3aec623b179d5dda82b2b38`.
Independent follow-up inspection closed the constructor-time finding with no
further concrete finding in that scope. The GPU repeat is still a throwaway
loader compatibility probe, not qualification of this whole installed adapter.

Before accepting a speed improvement: normally build/install a separate whole
Cachy prefix, freeze its CPU proof, run the same construction and ten-process
water CPU/GPU/oracle/displacement/repeat gate, and compare before/after artifacts
at unchanged tolerances. Record snapshot host memory/startup separately from
whole-energy time and tracked device memory. No Phoenix submission or H200
performance claim is implied.

#!/bin/bash
# Owned CLI boundary: manager/QCEngine Python stays in p4qca. Only this
# subprocess uses the sealed scientific interpreter and ordinary dependencies.
set -euo pipefail
: "${CCASP_NATIVE_ENV:?}" "${CCASP_NATIVE_BASE:?}" "${CCASP_NATIVE_CACHE:?}"
: "${CCASP_NATIVE_TRACE:?}" "${CCASP_NATIVE_HOME:?}"
[[ "$CCASP_NATIVE_ENV" = /storage/project/r-cs207-0/awallace43/runs/ccasp-native-s66-*/env ]]
[[ "$CCASP_NATIVE_BASE" = /storage/project/r-cs207-0/awallace43/miniconda/envs/p4cuest_sapt ]]
[[ "$CCASP_NATIVE_CACHE" = /storage/scratch1/3/awallace43/ccasp-native-*/* ]]
[[ "$CCASP_NATIVE_HOME" = /storage/scratch1/3/awallace43/ccasp-native-*/* ]]
[[ "$CCASP_NATIVE_TRACE" = /storage/project/r-cs207-0/awallace43/runs/ccasp-native-s66-*/* ]]
[[ -d "$CCASP_NATIVE_CACHE" && -d "$CCASP_NATIVE_TRACE" && -d "$CCASP_NATIVE_HOME" ]]
[[ -x /usr/bin/strace && -f "$CCASP_NATIVE_ENV/bin/psi4" ]]
unset PYTHONPATH PYTHONHOME PSIPATH PSIDATADIR LD_PRELOAD
export PATH="$CCASP_NATIVE_ENV/bin:$CCASP_NATIVE_BASE/bin:/usr/bin:/bin"
export LD_LIBRARY_PATH="$CCASP_NATIVE_ENV/lib:$CCASP_NATIVE_BASE/lib"
export HOME="$CCASP_NATIVE_HOME"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX="$CCASP_NATIVE_CACHE"
# Discovery and actual CLI execution are both traced without changing their
# stdout/stderr or exit status. Analysis must distinguish CLI/metadata
# processes from prohibited external scientific programs; zero children is
# not a valid claim for the QCEngine subprocess route.
exec /usr/bin/strace -f -qq -yy -s 4096 -e trace=%file,%process \
    -o "$CCASP_NATIVE_TRACE/psi4-$$.strace" \
    "$CCASP_NATIVE_ENV/bin/python" -I -B -X "pycache_prefix=$CCASP_NATIVE_CACHE" \
    "$CCASP_NATIVE_ENV/bin/psi4" "$@"

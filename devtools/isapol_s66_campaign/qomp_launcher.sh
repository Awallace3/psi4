#!/bin/bash
# Only the owned manager's QOMP CLI boundary; no source/native overlay.
set -euo pipefail
[[ "${CCASP_QOMP_SOURCE:?}" = /storage/project/r-cs207-0/awallace43/gits/psi4.ccasp-native-qomp-production148-v2 ]]
: "${CCASP_QOMP_BINDING:?}" "${CCASP_QOMP_BINDING_SHA256:?}" "${CCASP_NATIVE_CACHE:?}"
exec /storage/project/r-cs207-0/awallace43/miniconda/envs/p4qca/bin/python \
  -I -S -B -X "pycache_prefix=$CCASP_NATIVE_CACHE" \
  "$CCASP_QOMP_SOURCE/devtools/isapol_s66_campaign/qomp_execution.py" launcher "$@"

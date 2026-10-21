#!/usr/bin/env bash
# Source from a frozen build driver after compiler/module activation.
# No extraction/install/environment mutation; only process-local binding.
# Canonical SDK must already have been independently staged read-only.
cuest_sdk_select() {
    local source_root=$1 dependency_env=$2 evidence=$3
    export CUEST_SDK_ROOT
    CUEST_SDK_ROOT=$(realpath "${CUEST_SDK_ROOT:-${CPKS_CUEST_SDK:-/storage/project/r-cs207-0/awallace43/software/cuest-0.2.50-a06e8148}}")
    export CUEST_CUDA_RUNTIME_ROOT
    CUEST_CUDA_RUNTIME_ROOT=$(realpath "$dependency_env")
    export CUEST_CONTRACT="$source_root/cmake/cuest_sdk_contract.py"
    export CUEST_SDK_MANIFEST="$evidence/cuest-sdk.json"
    export LD_LIBRARY_PATH="$CUEST_SDK_ROOT/lib/13:$dependency_env/targets/x86_64-linux/lib:$dependency_env/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    "$dependency_env/bin/python" "$CUEST_CONTRACT" \
        --sdk-root "$CUEST_SDK_ROOT" --cuda-major 13 \
        --cuda-root "$CUEST_CUDA_RUNTIME_ROOT" --cudart "$dependency_env/lib/libcudart.so" \
        --output "$CUEST_SDK_MANIFEST"
    # This exact identity is checked in both outer and inner configurations.
    cuest_args=("-DcuEST_ROOT=$CUEST_SDK_ROOT" "-DcuEST_INCLUDE_DIR=$CUEST_SDK_ROOT/include"
                "-DcuEST_LIBRARY=$CUEST_SDK_ROOT/lib/13/libcuest.so"
                "-DcuEST_CUDA_RUNTIME_ROOT=$CUEST_CUDA_RUNTIME_ROOT"
                "-DcuEST_PARENT_IDENTITY=$CUEST_SDK_MANIFEST")
}

cuest_sdk_bind_core() {
    local dependency_env=$1 core=$2 evidence=$3
    "$dependency_env/bin/python" "$CUEST_CONTRACT" \
        --sdk-root "$CUEST_SDK_ROOT" --cuda-major 13 \
        --cuda-root "$CUEST_CUDA_RUNTIME_ROOT" --cudart "$dependency_env/lib/libcudart.so" \
        --expected "$CUEST_SDK_MANIFEST" --core "$core" \
        --output "$evidence/cuest-core-identity.json"
    export CUEST_CORE_IDENTITY="$evidence/cuest-core-identity.json"
}

cuest_sdk_reuse_core() {
    local dependency_env=$1 core=$2 evidence=$3 original=$4
    test -f "$original"
    "$dependency_env/bin/python" "$CUEST_CONTRACT" \
        --sdk-root "$CUEST_SDK_ROOT" --cuda-major 13 \
        --cuda-root "$CUEST_CUDA_RUNTIME_ROOT" --cudart "$dependency_env/lib/libcudart.so" \
        --expected "$CUEST_SDK_MANIFEST" --core "$core" --expected-core "$original" \
        --output "$evidence/cuest-core-identity.json"
    export CUEST_CORE_IDENTITY="$evidence/cuest-core-identity.json"
}

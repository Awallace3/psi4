#!/usr/bin/env bash
set -eo pipefail
harness=$1
campaign=$2
source /storage/project/r-cs207-0/awallace43/miniconda/etc/profile.d/conda.sh
conda activate p4qca
set -u
unset GCC_ROOT CPATH C_INCLUDE_PATH CPLUS_INCLUDE_PATH OBJC_INCLUDE_PATH LIBRARY_PATH PYTHONPATH PSI4_DATADIR
export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PATH="$CONDA_PREFIX/bin:/opt/slurm/current/bin:/usr/bin:/bin"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CMAKE_BUILD_PARALLEL_LEVEL=8
runtime="$campaign/runtime-pydantic2"
if [[ ! -x "$runtime/bin/python" ]]; then
    "$CONDA_PREFIX/bin/python" -m venv --system-site-packages "$runtime"
fi
# Private overlay only: p4qca's Pydantic 1/QCElemental 0.30 cannot load the
# QCElemental 0.50 required by these frozen production sources.
"$runtime/bin/python" -m pip install 'qcelemental==0.50.4' 'qcengine==0.50.0' 'pydantic==2.13.3' 'optking==0.4.2'
"$runtime/bin/python" -m pip freeze > "$campaign/runtime-pydantic2-packages.txt"
"$runtime/bin/python" -c 'import qcelemental, qcengine, pydantic; print(qcelemental.__version__, qcengine.__version__, pydantic.__version__)'
test -x "$("$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-cc" -print-prog-name=cc1)"
test -x "$("$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-c++" -print-prog-name=cc1plus)"
for arm in base head; do
    src="$campaign/sources/$arm"
    build="$campaign/builds/$arm"
    expected=$("$CONDA_PREFIX/bin/python" -c 'import json,sys; print(json.load(open(sys.argv[1]))[sys.argv[2]])' "$harness/config.json" "$arm")
    test "$(git -C "$src" rev-parse HEAD)" = "$expected"
    test -z "$(git -C "$src" status --porcelain)"
    cmake -S "$src" -B "$build" -G Ninja \
        -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$CONDA_PREFIX" \
        -DPython_EXECUTABLE="$runtime/bin/python" \
        -DCMAKE_C_COMPILER="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-cc" \
        -DCMAKE_CXX_COMPILER="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-c++" \
        -DENABLE_XHOST=OFF -DCMAKE_C_FLAGS="-march=x86-64-v3 -mtune=generic" \
        -DCMAKE_CXX_FLAGS="-march=x86-64-v3 -mtune=generic" \
        -DCMAKE_INSTALL_PREFIX="$campaign/install/$arm" \
        -DCMAKE_UNITY_BUILD=ON -DCMAKE_UNITY_BUILD_BATCH_SIZE=8 \
        -DENABLE_Einsums=OFF -DBUILD_TESTING=ON \
        -DCMAKE_INSIST_FIND_PACKAGE_Libint2=ON -DCMAKE_INSIST_FIND_PACKAGE_Libxc=ON \
        -DCMAKE_INSIST_FIND_PACKAGE_qcelemental=ON -DCMAKE_INSIST_FIND_PACKAGE_qcengine=ON
    cmake --build "$build" --parallel 8
    ctest --test-dir "$build" -R '^sapt-dft(1|2|-fdds-ooc)$' --output-on-failure -j 1
    PYTHONPATH="$build/stage/lib" PSI4_DATADIR="$build/stage/share/psi4" \
        "$runtime/bin/python" "$harness/validate_build.py" "$build" "$expected"
done

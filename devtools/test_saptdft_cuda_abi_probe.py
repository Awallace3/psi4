import importlib.util
import json
from pathlib import Path
import shutil
import subprocess

import pytest

spec = importlib.util.spec_from_file_location("abi_probe", Path(__file__).with_name("saptdft_cuda_abi_probe.py"))
abi = importlib.util.module_from_spec(spec)
spec.loader.exec_module(abi)


def test_header_flags_preserve_order_and_drop_compile_outputs():
    command = ('/bin/c++ -I/first -isystem /second -isystem/third -DVALUE=1 '
               '-B /tools --sysroot=/root -std=c++20 -O3 -flto -o file.o -c unity.cc')
    compiler, flags = abi.header_flags(command)
    assert compiler == "/bin/c++"
    assert flags == ["-I/first", "-isystem", "/second", "-isystem/third", "-DVALUE=1",
                     "-B", "/tools", "--sysroot=/root", "-std=c++20"]


@pytest.fixture
def fake_cuda(tmp_path, monkeypatch):
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("Native g++ needed for compile/link-only ABI regression")
    for key in ("CPATH", "C_INCLUDE_PATH", "CPLUS_INCLUDE_PATH", "OBJC_INCLUDE_PATH", "LIBRARY_PATH"):
        monkeypatch.delenv(key, raising=False)
    include13, include12 = tmp_path/"cuda13", tmp_path/"cuda12"
    include13.mkdir()
    include12.mkdir()
    for name in ("cublas_v2.h", "cusolverDn.h", "cuest.h"):
        (include13/name).write_text("")
    declaration = ('struct cudaDeviceProp { int value; }; typedef int cudaError_t;\n'
                   'extern "C" cudaError_t cudaGetDeviceProperties(cudaDeviceProp*, int);\n')
    (include13/"cuda_runtime.h").write_text("#define CUDART_VERSION 13030\n" + declaration)
    (include12/"cuda_runtime.h").write_text(
        "#define CUDART_VERSION 12060\n#define cudaGetDeviceProperties cudaGetDeviceProperties_v2\n"
        + declaration)
    runtime = tmp_path/"libfakecudart.so"
    source = 'extern "C" int cudaGetDeviceProperties(void*, int) { return 0; }\n'
    subprocess.run([compiler, "-shared", "-fPIC", "-x", "c++", "-", "-o", str(runtime)],
                   input=source, text=True, capture_output=True, check=True)
    return compiler, include13, include12, runtime


def test_real_gcc_cpath_contamination_red_then_green(fake_cuda, tmp_path, monkeypatch):
    compiler, include13, include12, runtime = fake_cuda
    flags = ["-I", str(include13), "-isystem", str(include13)]
    monkeypatch.setenv("CPATH", str(include12))
    red = abi.probe(compiler, flags, runtime, tmp_path/"red")
    assert not red["ok"] and red["cudart_version"] == 12060
    monkeypatch.delenv("CPATH")
    green = abi.probe(compiler, flags, runtime, tmp_path/"green")
    assert green["ok"] and green["cudart_version"] == 13030
    assert (tmp_path/"green/probe.so").exists()


def test_correct_version_but_unlinkable_runtime_is_rejected(fake_cuda, tmp_path):
    compiler, include13, _, runtime = fake_cuda
    subprocess.run([compiler, "-shared", "-fPIC", "-x", "c++", "-", "-o", str(runtime)],
                   input='extern "C" int unrelated() {return 0;}\n',
                   text=True, capture_output=True, check=True)
    result = abi.probe(compiler, ["-I", str(include13)], runtime, tmp_path/"bad-link")
    assert not result["ok"] and result["cudart_version"] == 13030
    assert "link" in result["error"]


def test_runtime_unity_command_is_selected(tmp_path):
    unity = tmp_path/"src/CMakeFiles/core.dir/Unity/unity_3_cxx.cxx"
    unity.parent.mkdir(parents=True)
    unity.write_text('#include "/source/psi4/src/cuest_runtime.cc"\n')
    database = tmp_path/"compile_commands.json"
    expected = dict(file=str(unity), command="/bin/c++ -I/path -c unit.cc", directory=str(tmp_path))
    database.write_text(json.dumps([expected]))
    assert abi.runtime_command(database) == expected

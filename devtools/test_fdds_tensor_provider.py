"""Compile the actual C++ supplier contract; storage is an explicit host model."""
from pathlib import Path
import shutil
import subprocess

import pytest


MODEL = r'''
#include "psi4/libsapt_solver/fdds_tensor_provider.h"
#include <algorithm>
#include <cassert>
#include <functional>
#include <numeric>
#include <vector>
using namespace psi::sapt;
class Memory final : public FDDSTensorProvider {
 public:
  int calls=0;
  std::map<std::string,std::vector<double>> data;
  void integral(std::string name, FDDSTensorShape dims) {
    register_integral(name,dims,FDDSTensorKind::RawCoulomb);
    auto& v=data[name]; v.resize(elements(dims));
    std::iota(v.begin(),v.end(),1.0);
  }
 private:
  void create_impl(const std::string& name,FDDSTensorShape dims) override {
    ++calls; data[name].resize(elements(dims));
  }
  void read_impl(const std::string& name,double* dst,FDDSTensorSlice s) override {
    ++calls; auto d=shape(name); auto stride=d[1]*d[2];
    std::copy(data[name].begin()+s[0]*stride,data[name].begin()+s[1]*stride,dst);
  }
  void write_impl(const std::string& name,const double* src,FDDSTensorSlice s) override {
    ++calls; auto d=shape(name); auto stride=d[1]*d[2];
    std::copy_n(src,(s[1]-s[0])*stride,data[name].begin()+s[0]*stride);
  }
};
void rejects(const std::function<void()>& f) {
  bool caught=false;
  try { f(); } catch (const std::runtime_error&) { caught=true; }
  assert(caught);
}
int main(int argc,char** argv) {
  assert(argc==2); const std::string mode=argv[1];
  Memory p; p.integral("raw",{3,2,5});
  if(mode=="layout") {
    for(auto dims : {FDDSTensorShape{3,7,11},FDDSTensorShape{7,3,11},
                      FDDSTensorShape{11,3,7},FDDSTensorShape{3,11,7}}) {
      Memory q; q.integral("input",dims);
      std::vector<double> b(q.elements(dims)+2,-999);
      for(size_t first=0;first<dims[0];first+=2) {
        auto last=std::min(first+2,dims[0]), n=(last-first)*dims[1]*dims[2];
        q.read("input",b.data(),b.size(),{first,last});
        for(size_t j=0;j<n;++j) assert(b[j]==1.0+first*dims[1]*dims[2]+j);
        assert(b.back()==-999);
      }
    }
  } else if(mode=="capacity") {
    double b[10]={};
    rejects([&]{p.read("raw",b,9,{0,1});});
    rejects([&]{p.read("raw",nullptr,10,{0,1});});
    rejects([&]{p.read("raw",b,10,{2,4});});
    rejects([&]{p.read("raw",b,10,{2,1});});
    rejects([&]{p.read("absent",b,10,{0,1});});
    assert(p.calls==0);
  } else if(mode=="overflow") {
    rejects([&]{p.create("bad",{std::numeric_limits<size_t>::max(),2,3});});
    rejects([&]{p.create("bad",{std::numeric_limits<size_t>::max()/sizeof(double)+1,1,1});});
    assert(p.calls==0);
  } else if(mode=="empty") {
    p.read("raw",nullptr,0,{3,3}); assert(p.calls==0);
    p.create("zero",{0,5,2}); p.read("zero",nullptr,0,{0,0});
    p.write("zero",nullptr,0,{0,0}); assert(p.calls==1);
  } else if(mode=="ownership") {
    double b[10]={};
    assert(p.kind("raw")==FDDSTensorKind::RawCoulomb);
    rejects([&]{p.write("raw",b,10,{0,1});});
    rejects([&]{p.create("raw",{3,2,5});});
    rejects([&]{p.create("",{1,1,1});});
    p.create("derived",{3,2,5});
    assert(p.kind("derived")==FDDSTensorKind::Derived);
    rejects([&]{p.create("derived",{3,2,5});});
    rejects([&]{p.write("derived",b,9,{0,1});});
    assert(p.calls==1);
  } else if(mode=="freshness") {
    p.create("derived",{3,2,5});
    std::vector<double> b(30), result(32,-1);
    for(int iteration=1;iteration!=4;++iteration) {
      std::iota(b.begin(),b.end(),iteration*100.0);
      auto original=b;
      p.write("derived",b.data(),20,{0,2});
      p.write("derived",b.data()+20,10,{2,3});
      assert(b==original);
      p.read("derived",result.data(),30,{0,3});
      assert(std::equal(b.begin(),b.end(),result.begin()));
      assert(result[30]==-1 && result[31]==-1);
    }
  } else return 2;
}
'''


@pytest.fixture(scope="module")
def supplier_model(tmp_path_factory):
    directory = tmp_path_factory.mktemp("fdds-supplier")
    cpp = directory / "model.cc"
    cpp.write_text(MODEL)
    compiler = shutil.which("c++")
    assert compiler, "C++ compiler required (this test must execute, not skip)"
    exe = directory / "model"
    root = Path(__file__).resolve().parents[1]
    subprocess.run([compiler, "-std=c++17", "-Wall", "-Wextra", "-Werror", "-O2",
                    "-I" + str(root / "psi4/src"), str(cpp), "-o", str(exe)],
                   check=True, capture_output=True, text=True)
    return exe


@pytest.mark.parametrize("mode", ["layout", "capacity", "overflow", "empty", "ownership", "freshness"])
def test_supplier_contract(supplier_model, mode):
    subprocess.run([str(supplier_model), mode], check=True, timeout=10)

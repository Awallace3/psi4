// Small algebra fixture for extracted native PCG/action code, NOT a Psi4 build.
#include <algorithm>
#include <cmath>
#include <ctime>
#include <iostream>
#include <iomanip>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#define PSIEXCEPTION(message) std::runtime_error(message)
struct Matrix {
    int rows, cols;
    std::vector<double> data;
    std::vector<double*> pointers;
    Matrix(int r, int c) : rows(r), cols(c), data(r*c), pointers(r) {}
    int nrow() const { return rows; }
    int ncol() const { return cols; }
    int nirrep() const { return 1; }
    double get(int i, int j) const { return data[i*cols+j]; }
    double** pointer() { for (int i=0; i<rows; ++i) pointers[i]=data.data()+i*cols; return pointers.data(); }
    std::shared_ptr<Matrix> clone() { auto m=std::make_shared<Matrix>(rows,cols); m->data=data; return m; }
    void copy(std::shared_ptr<Matrix> m) { data=m->data; }
    void zero() { std::fill(data.begin(),data.end(),0.); }
    void scale(double a) { for (auto& x:data) x*=a; }
    double vector_dot(std::shared_ptr<Matrix> m) {
        double sum=0.; for (size_t i=0; i<data.size(); ++i) sum+=data[i]*m->data[i]; return sum;
    }
    void subtract(std::shared_ptr<Matrix> m) { for (size_t i=0; i<data.size(); ++i) data[i]-=m->data[i]; }
};
using SharedMatrix=std::shared_ptr<Matrix>;
struct Vector {
    std::vector<double> data;
    explicit Vector(int n) : data(n) {}
    int dim() { return data.size(); }
    double* pointer() { return data.data(); }
};
struct JK {};
namespace scf {
struct HF {
    std::vector<double> hessian;
    double offset=0.;
    std::vector<SharedMatrix> cphf_Hx(std::vector<SharedMatrix> x) {
        auto y=x[0]->clone(); const size_t n=y->data.size();
        for (size_t i=0; i<n; ++i) {
            y->data[i]=offset;
            for (size_t j=0; j<n; ++j) y->data[i]-=hessian[i*n+j]*x[0]->data[j];
        }
        return {y};
    }
};
}
struct Printer { template<class... T> void Printf(const char*,T...) {} } printer;
auto outfile=&printer;
inline void C_DAXPY(int n,double a,double* x,int dx,double* y,int dy) {
    for (int i=0;i<n;++i) y[i*dy]+=a*x[i*dx];
}
inline double C_DDOT(int n,double* x,int dx,double* y,int dy) {
    double sum=0.; for (int i=0;i<n;++i) sum+=x[i*dx]*y[i*dy]; return sum;
}
inline void C_DSCAL(int n,double a,double* x,int dx) { for (int i=0;i<n;++i) x[i*dx]*=a; }

// Test input/output only. The class and numerical methods below this fixture
// are extracted verbatim from fisapt.h/fisapt.cc by test_fsapt_cpks_find.py.
#define protected public

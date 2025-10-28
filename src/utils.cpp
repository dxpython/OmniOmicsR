#include <Rcpp.h>
using namespace Rcpp;

// [[Rcpp::export]]
NumericMatrix fast_row_z(NumericMatrix x) {
  int n = x.nrow(), p = x.ncol();
  NumericMatrix out(n, p);
  for (int i = 0; i < n; ++i) {
    double sum = 0.0, sumsq = 0.0; int cnt = 0;
    for (int j = 0; j < p; ++j) {
      double v = x(i, j);
      if (R_finite(v)) { sum += v; sumsq += v * v; cnt++; }
    }
    double mu = cnt ? sum / cnt : 0.0;
    double sd = cnt > 1 ? sqrt((sumsq - cnt * mu * mu) / (cnt - 1)) : 1.0;
    if (sd == 0) sd = 1.0;
    for (int j = 0; j < p; ++j) {
      double v = x(i, j);
      out(i, j) = R_finite(v) ? (v - mu) / sd : NA_REAL;
    }
  }
  return out;
}

// [[Rcpp::export]]
NumericMatrix fast_col_z(NumericMatrix x) {
  int n = x.nrow(), p = x.ncol();
  NumericMatrix out(n, p);
  for (int j = 0; j < p; ++j) {
    double sum = 0.0, sumsq = 0.0; int cnt = 0;
    for (int i = 0; i < n; ++i) {
      double v = x(i, j);
      if (R_finite(v)) { sum += v; sumsq += v * v; cnt++; }
    }
    double mu = cnt ? sum / cnt : 0.0;
    double sd = cnt > 1 ? sqrt((sumsq - cnt * mu * mu) / (cnt - 1)) : 1.0;
    if (sd == 0) sd = 1.0;
    for (int i = 0; i < n; ++i) {
      double v = x(i, j);
      out(i, j) = R_finite(v) ? (v - mu) / sd : NA_REAL;
    }
  }
  return out;
}


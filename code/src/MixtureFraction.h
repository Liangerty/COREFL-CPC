#pragma once

#include "Define.h"
#include "gxl_lib/Matrix.hpp"
#include "gxl_lib/Array.hpp"
#include "BoundCond.h"
#include "ChemData.h"

namespace cfd {

struct MixtureFraction {
  virtual ~MixtureFraction() = default;

  int n_spec{0};
  real beta_f{0}, beta_o{0};
  real beta_diff{0};
  real z_st{0}, f_zst{0};
  const gxl::MatrixDyn<int> &elem_comp;
  const std::vector<real> &mw;

  explicit MixtureFraction(Inflow &fuel, Inflow &oxidizer, const Species &chem_data);

  virtual real compute_mixture_fraction(std::vector<real> &yk) = 0;
};

class BilgerH final : public MixtureFraction {
  real nuh_mwh{0}, half_nuo_mwo{0};
  int elem_label[2]{0, 1};
public:
  explicit BilgerH(Inflow &fuel, Inflow &oxidizer, const Species &spec, int myid = 0);

  real compute_mixture_fraction(std::vector<real> &yk) override;

private:
  [[nodiscard]] real compute_coupling_function(real z_h, real z_o) const;
};

class BilgerCH final : public MixtureFraction {
  real nuc_mwc{0}, nuh_mwh{0}, half_nuo_mwo{0};
  int elem_label[3]{0, 1, 2};
public:
  explicit BilgerCH(Inflow &fuel, Inflow &oxidizer, const Species &spec, int myid = 0);

  real compute_mixture_fraction(std::vector<real> &yk) override;

private:
  [[nodiscard]] real compute_coupling_function(real z_c, real z_h, real z_o) const;
};

void
acquire_mixture_fraction_expression(const Species &spec, const real *fuel, const real *oxidizer, Parameter &parameter);

} // cfd

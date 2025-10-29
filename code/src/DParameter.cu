#include "DParameter.cuh"
#include "ChemData.h"
#include <filesystem>
#include "Constants.h"

cfd::DParameter::DParameter(Parameter &parameter, const Species &species, Reaction *reaction) :
  myid{parameter.get_int("myid")},
  dim{parameter.get_int("dimension")}, problem_type{parameter.get_int("problem_type")},
  n_var{parameter.get_int("n_var")}, n_scalar_transported{parameter.get_int("n_scalar_transported")},
  i_fl{parameter.get_int("i_fl")}, i_fl_cv{parameter.get_int("i_fl_cv")},
  i_turb_cv{parameter.get_int("i_turb_cv")}, inviscid_scheme{parameter.get_int("inviscid_scheme")},
  reconstruction{parameter.get_int("reconstruction")}, limiter{parameter.get_int("limiter")},
  viscous_scheme{parameter.get_int("viscous_order")}, positive_preserving{parameter.get_bool("positive_preserving")},
  gradPInDiffusionFlux{parameter.get_bool("gradPInDiffusionFlux")},
  entropy_fix_factor{parameter.get_real("entropy_fix_factor")}, fixed_dt{parameter.get_bool("fixed_time_step")},
  rans_model{parameter.get_int("RANS_model")}, turb_implicit{parameter.get_int("turb_implicit")},
  compressibility_correction{parameter.get_int("compressibility_correction")},
  chemSrcMethod{parameter.get_int("chemSrcMethod")},
  Pr(parameter.get_real("prandtl_number")), cfl(parameter.get_real("cfl")),
  Prt(parameter.get_real("turbulent_prandtl_number")),
  Sct(parameter.get_real("turbulent_schmidt_number")),
  rho_ref{parameter.get_real("rho_inf")},
  a_ref2{parameter.get_real("speed_of_sound") * parameter.get_real("speed_of_sound")},
  v_ref{parameter.get_real("v_inf")}, T_ref{parameter.get_real("T_inf")}, mach_ref{parameter.get_real("M_inf")},
  p_ref{parameter.get_real("p_inf")}, weno_eps_scale{
    parameter.get_real("rho_inf") * parameter.get_real("v_inf") * parameter.get_real("rho_inf") *
    parameter.get_real("v_inf")
  }, perform_spanwise_average{parameter.get_bool("perform_spanwise_average")}
/*,sponge_layer{parameter.get_bool("sponge_layer")}, sponge_function{parameter.get_int("sponge_function")},
sponge_iter{parameter.get_int("sponge_iter")}, spongeXMinusStart{parameter.get_real("spongeXMinusStart")},
spongeXMinusEnd{parameter.get_real("spongeXMinusEnd")}, spongeXPlusStart{parameter.get_real("spongeXPlusStart")},
spongeXPlusEnd{parameter.get_real("spongeXPlusEnd")}, spongeYMinusStart{parameter.get_real("spongeYMinusStart")},
spongeYMinusEnd{parameter.get_real("spongeYMinusEnd")}, spongeYPlusStart{parameter.get_real("spongeYPlusStart")},
spongeYPlusEnd{parameter.get_real("spongeYPlusEnd")}, spongeZMinusStart{parameter.get_real("spongeZMinusStart")},
spongeZMinusEnd{parameter.get_real("spongeZMinusEnd")}, spongeZPlusStart{parameter.get_real("spongeZPlusStart")},
spongeZPlusEnd{parameter.get_real("spongeZPlusEnd")}*/ {
  if (parameter.get_int("myid") == 0) {
    if (inviscid_scheme == 51 || inviscid_scheme == 52 || inviscid_scheme == 71 || inviscid_scheme == 72)
      printf("\t->-> %-20e : WENO scale factor\n", weno_eps_scale);
  }

  if (parameter.get_bool("fixed_time_step")) {
    dt = parameter.get_real("dt");
  }

  // if (parameter.get_string("hybrid_inviscid_scheme") != "NO") {
  shock_sensor = parameter.get_int("shock_sensor");
  sensor_eps = parameter.get_real("shockSensor_epsilon");
  sensor_threshold = parameter.get_real("shockSensor_threshold");
  // }

  const auto &spec = species;
  n_spec = spec.n_spec;
  n_scalar = parameter.get_int("n_scalar");
  if (reaction != nullptr) {
    n_reac = reaction->n_reac;
  }
  n_ps = parameter.get_int("n_ps");
  i_ps = parameter.get_int("i_ps");
  i_ps_cv = parameter.get_int("i_ps_cv");
  if (n_ps > 0) {
    cudaMalloc(&sc_ps, n_ps * sizeof(real));
    cudaMemcpy(sc_ps, parameter.get_real_array("sc_passive_scalar").data(), n_ps * sizeof(real),
               cudaMemcpyHostToDevice);
    cudaMalloc(&sct_ps, n_ps * sizeof(real));
    cudaMemcpy(sct_ps, parameter.get_real_array("sct_passive_scalar").data(), n_ps * sizeof(real),
               cudaMemcpyHostToDevice);
  }

  // species info
  auto mem_sz = n_spec * sizeof(real);
  // cudaMalloc(&mw, mem_sz);
  // cudaMemcpy(mw, spec.mw.data(), mem_sz, cudaMemcpyHostToDevice);
  std::vector<real> imw_vec(n_spec), Rl(n_spec);
  for (int l = 0; l < n_spec; ++l) {
    imw_vec[l] = 1.0 / spec.mw[l];
    Rl[l] = R_u / spec.mw[l];
  }
  cudaMalloc(&imw, mem_sz);
  cudaMemcpy(imw, imw_vec.data(), mem_sz, cudaMemcpyHostToDevice);
  cudaMalloc(&gas_const, mem_sz);
  cudaMemcpy(gas_const, Rl.data(), mem_sz, cudaMemcpyHostToDevice);
  #ifdef HighTempMultiPart
  cudaMalloc(&n_temperature_range, n_spec * sizeof(int));
  cudaMemcpy(n_temperature_range, spec.n_temperature_range.data(), n_spec * sizeof(int), cudaMemcpyHostToDevice);
  int n_ranges = 2;
  for (int l = 0; l < n_spec; ++l) {
    n_ranges = std::max(spec.n_temperature_range[l], n_ranges);
  }
  temperature_cuts.init_with_size(n_spec, n_ranges + 1);
  cudaMemcpy(temperature_cuts.data(), spec.temperature_range.data(), temperature_cuts.size() * sizeof(real),
             cudaMemcpyHostToDevice);
  therm_poly_coeff.allocate_memory(7, n_ranges, n_spec, 0);
  cudaMemcpy(therm_poly_coeff.data(), spec.therm_poly_coeff.data(), sizeof(real) * therm_poly_coeff.size(),
             cudaMemcpyHostToDevice);
  #else
  high_temp_coeff.init_with_size(n_spec, 7);
  cudaMemcpy(high_temp_coeff.data(), spec.high_temp_coeff.data(), high_temp_coeff.size() * sizeof(real),
             cudaMemcpyHostToDevice);
  low_temp_coeff.init_with_size(n_spec, 7);
  cudaMemcpy(low_temp_coeff.data(), spec.low_temp_coeff.data(), low_temp_coeff.size() * sizeof(real),
             cudaMemcpyHostToDevice);
  cudaMalloc(&t_low, mem_sz);
  cudaMalloc(&t_mid, mem_sz);
  cudaMalloc(&t_high, mem_sz);
  cudaMemcpy(t_low, spec.t_low.data(), mem_sz, cudaMemcpyHostToDevice);
  cudaMemcpy(t_mid, spec.t_mid.data(), mem_sz, cudaMemcpyHostToDevice);
  cudaMemcpy(t_high, spec.t_high.data(), mem_sz, cudaMemcpyHostToDevice);
  #endif
  cudaMalloc(&geometry, mem_sz);
  cudaMemcpy(geometry, spec.geometry.data(), mem_sz, cudaMemcpyHostToDevice);
  cudaMalloc(&LJ_potent_inv, mem_sz);
  cudaMemcpy(LJ_potent_inv, spec.LJ_potent_inv.data(), mem_sz, cudaMemcpyHostToDevice);
  cudaMalloc(&vis_coeff, mem_sz);
  cudaMemcpy(vis_coeff, spec.vis_coeff.data(), mem_sz, cudaMemcpyHostToDevice);
  WjDivWi_to_One4th.init_with_size(n_spec, n_spec);
  cudaMemcpy(WjDivWi_to_One4th.data(), spec.WjDivWi_to_One4th.data(), WjDivWi_to_One4th.size() * sizeof(real),
             cudaMemcpyHostToDevice);
  sqrt_WiDivWjPl1Mul8.init_with_size(n_spec, n_spec);
  cudaMemcpy(sqrt_WiDivWjPl1Mul8.data(), spec.sqrt_WiDivWjPl1Mul8.data(),
             sqrt_WiDivWjPl1Mul8.size() * sizeof(real), cudaMemcpyHostToDevice);
  binary_diffusivity_coeff.init_with_size(n_spec, n_spec);
  cudaMemcpy(binary_diffusivity_coeff.data(), spec.binary_diffusivity_coeff.data(),
             binary_diffusivity_coeff.size() * sizeof(real), cudaMemcpyHostToDevice);
  kb_over_eps_jk.init_with_size(n_spec, n_spec);
  cudaMemcpy(kb_over_eps_jk.data(), spec.kb_over_eps_jk.data(),
             kb_over_eps_jk.size() * sizeof(real), cudaMemcpyHostToDevice);
  cudaMalloc(&ZRotF298, mem_sz);
  cudaMemcpy(ZRotF298, spec.ZRotF298.data(), mem_sz, cudaMemcpyHostToDevice);
  Sc = parameter.get_real("schmidt_number");

  // reaction info
  if (n_reac > 0) {
    hardCodedMech = parameter.get_int("hardCodedMech");
    cudaMalloc(&reac_type, n_reac * sizeof(int));
    cudaMemcpy(reac_type, reaction->label.data(), n_reac * sizeof(int), cudaMemcpyHostToDevice);
    cudaMalloc(&rev_type, n_reac * sizeof(int));
    cudaMemcpy(rev_type, reaction->rev_type.data(), n_reac * sizeof(int), cudaMemcpyHostToDevice);
    stoi_f.init_with_size(n_reac, n_spec);
    cudaMemcpy(stoi_f.data(), reaction->stoi_f.data(), stoi_f.size() * sizeof(int), cudaMemcpyHostToDevice);
    stoi_b.init_with_size(n_reac, n_spec);
    cudaMemcpy(stoi_b.data(), reaction->stoi_b.data(), stoi_b.size() * sizeof(int), cudaMemcpyHostToDevice);
    mem_sz = n_reac * sizeof(real);
    cudaMalloc(&reac_order, n_reac * sizeof(int));
    cudaMemcpy(reac_order, reaction->order.data(), n_reac * sizeof(int), cudaMemcpyHostToDevice);
    cudaMalloc(&A, mem_sz);
    cudaMemcpy(A, reaction->A.data(), mem_sz, cudaMemcpyHostToDevice);
    cudaMalloc(&b, mem_sz);
    cudaMemcpy(b, reaction->b.data(), mem_sz, cudaMemcpyHostToDevice);
    cudaMalloc(&Ea, mem_sz);
    cudaMemcpy(Ea, reaction->Ea.data(), mem_sz, cudaMemcpyHostToDevice);
    cudaMalloc(&A2, mem_sz);
    cudaMemcpy(A2, reaction->A2.data(), mem_sz, cudaMemcpyHostToDevice);
    cudaMalloc(&b2, mem_sz);
    cudaMemcpy(b2, reaction->b2.data(), mem_sz, cudaMemcpyHostToDevice);
    cudaMalloc(&Ea2, mem_sz);
    cudaMemcpy(Ea2, reaction->Ea2.data(), mem_sz, cudaMemcpyHostToDevice);
    third_body_coeff.init_with_size(n_reac, n_spec);
    cudaMemcpy(third_body_coeff.data(), reaction->third_body_coeff.data(), third_body_coeff.size() * sizeof(real),
               cudaMemcpyHostToDevice);
    cudaMalloc(&troe_alpha, mem_sz);
    cudaMemcpy(troe_alpha, reaction->troe_alpha.data(), mem_sz, cudaMemcpyHostToDevice);
    cudaMalloc(&troe_t3, mem_sz);
    cudaMemcpy(troe_t3, reaction->troe_t3.data(), mem_sz, cudaMemcpyHostToDevice);
    cudaMalloc(&troe_t1, mem_sz);
    cudaMemcpy(troe_t1, reaction->troe_t1.data(), mem_sz, cudaMemcpyHostToDevice);
    cudaMalloc(&troe_t2, mem_sz);
    cudaMemcpy(troe_t2, reaction->troe_t2.data(), mem_sz, cudaMemcpyHostToDevice);
  }

  if (parameter.get_bool("if_collect_statistics")) {
    n_reyAve = static_cast<int>(parameter.get_int_array("reyAveVarIndex").size());
    cudaMalloc(&reyAveVarIndex, n_reyAve * sizeof(int));
    cudaMemcpy(reyAveVarIndex, parameter.get_int_array("reyAveVarIndex").data(), n_reyAve * sizeof(int),
               cudaMemcpyHostToDevice);
    n_reyAveScalar = static_cast<int>(parameter.get_int_array("reyAveScalarIndex").size());
    cudaMalloc(&reyAveScalarIndex, n_reyAveScalar * sizeof(int));
    cudaMemcpy(reyAveScalarIndex, parameter.get_int_array("reyAveScalarIndex").data(), n_reyAveScalar * sizeof(int),
               cudaMemcpyHostToDevice);
    rho_p_correlation = parameter.get_bool("rho_p_correlation");
    stat_tke_budget = parameter.get_bool("stat_tke_budget");
    stat_scalar_fluc_budget = parameter.get_bool("stat_scalar_fluc_budget");
    stat_species_dissipation_rate = parameter.get_bool("stat_species_dissipation_rate");
    stat_species_velocity_correlation = parameter.get_bool("stat_species_velocity_correlation");
    n_species_stat = parameter.get_int("n_species_stat");
    cudaMalloc(&specStatIndex, n_species_stat * sizeof(int));
    cudaMemcpy(specStatIndex, parameter.get_int_array("species_stat_index").data(), n_species_stat * sizeof(int),
               cudaMemcpyHostToDevice);
  }

  if (parameter.get_string("canonical_problem") == "jicf") {
    n_jet = parameter.get_int("n_jet");
    jet_radius = parameter.get_real("jet_radius");
    if (n_jet > 0) {
      cudaMalloc(&xc_jet, n_jet * sizeof(real));
      cudaMemcpy(xc_jet, parameter.get_real_array("xc_jet").data(), n_jet * sizeof(real), cudaMemcpyHostToDevice);
      cudaMalloc(&zc_jet, n_jet * sizeof(real));
      cudaMemcpy(zc_jet, parameter.get_real_array("zc_jet").data(), n_jet * sizeof(real), cudaMemcpyHostToDevice);

      // jet info
      std::vector<real> jet_uh(n_jet), jet_vh(n_jet), jet_wh(n_jet), jet_Th(n_jet), jet_ph(n_jet), jet_rhoh(n_jet);
      gxl::MatrixDyn<real> jet_svh;
      jet_svh.resize(n_jet, n_spec);
      for (int i = 0; i < n_jet; i++) {
        auto struct_name = "jet-" + std::to_string(i);
        auto info = parameter.get_struct(struct_name);
        auto jet1_mach = std::get<real>(info.at("mach"));
        auto jet1_u = std::get<real>(info.at("u"));
        auto jet1_v = std::get<real>(info.at("v"));
        auto jet1_w = std::get<real>(info.at("w"));
        auto jet1_T = std::get<real>(info.at("temperature"));
        auto jet1_p = std::get<real>(info.at("pressure"));
        real jet1_rho;
        auto jet1_sv = new real[n_spec];
        memset(jet1_sv, 0, n_spec * sizeof(real));
        if (n_spec > 0) {
          // const auto& spec_list = spec.spec_list;
          real mw_inv = 0;
          for (const auto &[name, idx]: spec.spec_list) {
            if (info.find(name) != info.cend()) {
              jet1_sv[idx] = std::get<real>(info.at(name));
              // printf("%s = %e\n", name.c_str(), jet1_sv[idx]);
            }
          }
          for (int l = 0; l < n_spec; ++l) {
            mw_inv += jet1_sv[l] / spec.mw[l];
          }
          jet1_rho = jet1_p / (R_u * mw_inv * jet1_T);
          // printf("rho = %e\n", jet1_rho);
          std::vector<real> cpi(n_spec, 0);
          spec.compute_cp(jet1_T, cpi.data());
          real cp{0}, cv{0};
          for (int l = 0; l < n_spec; ++l) {
            cp += jet1_sv[l] * cpi[l];
            cv += jet1_sv[l] * (cpi[l] - R_u / spec.mw[l]);
          }
          real gamma = cp / cv; // specific heat ratio
          real c = sqrt(gamma * R_u * jet1_T * mw_inv);
          // printf("cp = %f, cv = %f, gamma = %f, c=%f\n", cp, cv, gamma, c);
          jet1_u *= jet1_mach * c;
          jet1_v *= jet1_mach * c;
          jet1_w *= jet1_mach * c;
        } else {
          jet1_rho = jet1_p / (R_air * jet1_T);
          real c = sqrt(gamma_air * R_air * jet1_T);
          jet1_u *= jet1_mach * c;
          jet1_v *= jet1_mach * c;
          jet1_w *= jet1_mach * c;
        }
        jet_uh[i] = jet1_u;
        jet_vh[i] = jet1_v;
        jet_wh[i] = jet1_w;
        jet_Th[i] = jet1_T;
        jet_ph[i] = jet1_p;
        jet_rhoh[i] = jet1_rho;
        for (int l = 0; l < n_spec; ++l) {
          jet_svh(i, l) = jet1_sv[l];
        }
        // printf("rho=%e, u=%e, v=%e, w=%e, T=%e, p=%e\n", jet1_rho, jet1_u, jet1_v, jet1_w, jet1_T, jet1_p);
      }
      cudaMalloc(&jet_u, n_jet * sizeof(real));
      cudaMemcpy(jet_u, jet_uh.data(), n_jet * sizeof(real), cudaMemcpyHostToDevice);
      cudaMalloc(&jet_v, n_jet * sizeof(real));
      cudaMemcpy(jet_v, jet_vh.data(), n_jet * sizeof(real), cudaMemcpyHostToDevice);
      cudaMalloc(&jet_w, n_jet * sizeof(real));
      cudaMemcpy(jet_w, jet_wh.data(), n_jet * sizeof(real), cudaMemcpyHostToDevice);
      cudaMalloc(&jet_T, n_jet * sizeof(real));
      cudaMemcpy(jet_T, jet_Th.data(), n_jet * sizeof(real), cudaMemcpyHostToDevice);
      cudaMalloc(&jet_p, n_jet * sizeof(real));
      cudaMemcpy(jet_p, jet_ph.data(), n_jet * sizeof(real), cudaMemcpyHostToDevice);
      cudaMalloc(&jet_rho, n_jet * sizeof(real));
      cudaMemcpy(jet_rho, jet_rhoh.data(), n_jet * sizeof(real), cudaMemcpyHostToDevice);
      jet_sv.init_with_size(n_jet, n_spec);
      cudaMemcpy(jet_sv.data(), jet_svh.data(), n_jet * n_spec * sizeof(real), cudaMemcpyHostToDevice);
    }
  }

  // If mixing layer and multi-component, we need the mixture fraction info.
  if (problem_type == 1 && n_spec > 0) {
    beta_diff_inv = parameter.get_real("beta_diff_inv");
    beta_o = parameter.get_real("beta_o");
    nuc_mwc_inv = parameter.get_real("nuc_mwc_inv");
    nuh_mwh_inv = parameter.get_real("nuh_mwh_inv");
    half_nuo_mwo_inv = parameter.get_real("half_nuo_mwo_inv");
  }

  // the following parameters have been computed in "write_reference_state".
  if (problem_type == 1) {
    convective_velocity = parameter.get_real("convective_velocity");
    delta_omega0 = parameter.get_real("delta_omega");
    if (int i = parameter.get_int("characteristic_velocity_ml"); i == 0) {
      v_char = parameter.get_real("convective_velocity");
    } else {
      v_char = parameter.get_real("DeltaU");
    }
    delta_u = parameter.get_real("DeltaU");
  } else {
    v_char = parameter.get_real("v_inf");
  }

  fluctuation_form = parameter.get_int("fluctuation_form");
  fluctuation_intensity = parameter.get_real("fluctuation_intensity");
  N_spanwise_wave = parameter.get_int("N_spanwise_waves");
  x0_fluc = parameter.get_real("x0_fluc");
  y0_fluc = parameter.get_real("y0_fluc");
  z0_fluc = parameter.get_real("z0_fluc");

  memset(limit_flow.ll, 0, sizeof(real) * LimitFlow::max_n_var);
  memset(limit_flow.ul, 0, sizeof(real) * LimitFlow::max_n_var);
  memset(limit_flow.sv_inf, 0, sizeof(real) * (MAX_SPEC_NUMBER + 2));
  // density limits
  limit_flow.ll[0] = 1e-6 * parameter.get_real("rho_inf");
  limit_flow.ul[0] = 1e+3 * parameter.get_real("rho_inf");
  for (int l = 1; l < 4; ++l) {
    real vRef{parameter.get_real("v_inf")};
    if (abs(vRef) < 1) {
      vRef = parameter.get_real("speed_of_sound");
    }
    limit_flow.ll[l] = -1000 * vRef;
    limit_flow.ul[l] = 1000 * vRef;
  }
  // pressure limits
  limit_flow.ll[4] = 1e-6 * parameter.get_real("p_inf");
  limit_flow.ul[4] = 1e+3 * parameter.get_real("p_inf");
  if (rans_model == 2) {
    // SST model
    limit_flow.ul[5] = std::numeric_limits<real>::max();
    limit_flow.ul[6] = std::numeric_limits<real>::max();
  }
  auto &sv_inf{parameter.get_real_array("sv_inf")};
  for (int l = 0; l < n_scalar; ++l) {
    limit_flow.sv_inf[l] = sv_inf[l];
  }

  // if (parameter.get_bool("sponge_layer")) {
  //   spongeX = parameter.get_int("spongeX");
  //   spongeY = parameter.get_int("spongeY");
  //   spongeZ = parameter.get_int("spongeZ");
  //   if (parameter.get_int("n_scalar") > 0) {
  //     cudaMalloc(&sponge_scalar_iter, n_scalar * sizeof(int));
  //     cudaMemcpy(sponge_scalar_iter, parameter.get_int_array("sponge_scalar_iter").data(), n_scalar * sizeof(int),
  //                cudaMemcpyHostToDevice);
  //   }
  //   if (spongeX == 1 || spongeX == 3) {
  //     sponge_sigma0 = parameter.get_real("spongeCoefficient") * v_char / (spongeXMinusStart - spongeXMinusEnd);
  //     printf("sponge_sigma0=%e\n", sponge_sigma0);
  //   }
  //   if (spongeX == 2 || spongeX == 3) {
  //     sponge_sigma1 = parameter.get_real("spongeCoefficient") * v_char / (spongeXPlusEnd - spongeXPlusStart);
  //     printf("sponge_sigma1=%e\n", sponge_sigma1);
  //   }
  //   if (spongeY == 1 || spongeY == 3) {
  //     sponge_sigma2 = parameter.get_real("spongeCoefficient") * v_char / (spongeYMinusStart - spongeYMinusEnd);
  //     printf("sponge_sigma2=%e\n", sponge_sigma2);
  //   }
  //   if (spongeY == 2 || spongeY == 3) {
  //     sponge_sigma3 = parameter.get_real("spongeCoefficient") * v_char / (spongeYPlusEnd - spongeYPlusStart);
  //     printf("sponge_sigma3=%e\n", sponge_sigma3);
  //   }
  //   if (spongeZ == 1 || spongeZ == 3) {
  //     sponge_sigma4 = parameter.get_real("spongeCoefficient") * v_char / (spongeZMinusStart - spongeZMinusEnd);
  //     printf("sponge_sigma4=%e\n", sponge_sigma4);
  //   }
  //   if (spongeZ == 2 || spongeZ == 3) {
  //     sponge_sigma5 = parameter.get_real("spongeCoefficient") * v_char / (spongeZPlusEnd - spongeZPlusStart);
  //     printf("sponge_sigma5=%e\n", sponge_sigma5);
  //   }
  // }
}

//cfd::DParameter::~DParameter() {
//  cudaFree(mw);
//  cudaFree(t_low);
//  cudaFree(t_mid);
//  cudaFree(t_high);
//  cudaFree(LJ_potent_inv);
//  cudaFree(vis_coeff);
//  cudaFree(reac_type);
//  cudaFree(reac_order);
//  cudaFree(A);
//  cudaFree(b);
//  cudaFree(Ea);
//  cudaFree(A2);
//  cudaFree(b2);
//  cudaFree(Ea2);
//  cudaFree(troe_alpha);
//  cudaFree(troe_t3);
//  cudaFree(troe_t1);
//  cudaFree(troe_t2);
//  cudaFree(mix_frac);
//}
__global__ void cfd::update_dt_global(DParameter *param, real dt) {
  param->dt = dt;
}

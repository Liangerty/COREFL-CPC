#pragma once

namespace cfd {
enum class TurbSimLevel {
  Lam,
  RANS,
  DES,
  LES,
  DNS
};

enum class TurbMethodLabel {
  Lam,
  SST,
  SA,
  DES,
  LES,
  DNS
};

template<typename T>
struct TurbMethod {
  // If this simulation method is for laminar or turbulent. All methods from RANS to DNS are all false.
  static constexpr bool isLaminar = true;
  // If this model requires the mut variable. For now, RANS, DES, and LES all need it.
  static constexpr bool hasMut = false;
  // Which kind of turbulent simulation method is this, which may be Lam, RANS, DES, LES, or DNS.
  static constexpr TurbSimLevel type = TurbSimLevel::Lam;
  // The concrete label of this method, which may be Lam, SST, SA, DES, LES, or DNS.
  static constexpr TurbMethodLabel label = TurbMethodLabel::Lam;
  // If we need to compute distance to wall in this method
  static constexpr bool needWallDistance = false;
  // If this method has implicit treatment for the source term
  static constexpr bool hasImplicitTreat = false;
};

struct Laminar {
};

}
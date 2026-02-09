#pragma once

#include <string>

namespace cfd {
class Element {
  std::string name;
public:
  explicit Element(std::string a) : name{std::move(a)} {
    for (auto &ch: name) {
      ch = std::toupper(ch);
    }
  }

  [[nodiscard]] double get_atom_weight() const;
};
}

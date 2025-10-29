#include "MyString.h"
#include <fstream>
#include <fmt/core.h>
#include <algorithm>

void gxl::trim_left(std::string &string) {
  const auto start = string.find_first_not_of(whiteSpace);
  if (start != std::string::npos) {
    string = string.substr(start);
  } else {
    string = "";
  }
}

std::istream &gxl::getline(std::ifstream &file, std::string &input, Case u_l) {
  std::getline(file, input);
  trim_left(input);
  if (u_l == Case::upper) {
    for (auto &ch: input) {
      ch = std::toupper(ch);
    }
  } else if (u_l == Case::lower) {
    for (auto &ch: input) {
      ch = std::tolower(ch);
    }
  }
  return file;
}

void gxl::to_stringstream(const std::string &input, std::istringstream &line) {
  line.clear();
  line.str(input);
}

std::istream &gxl::getline_to_stream(std::ifstream &file, std::string &input,
                                     std::istringstream &line, Case u_l) {
  std::getline(file, input);
  trim_left(input);
  if (u_l == Case::upper) {
    for (auto &ch: input) {
      ch = std::toupper(ch);
    }
  } else if (u_l == Case::lower) {
    for (auto &ch: input) {
      ch = std::tolower(ch);
    }
  }
  line.clear();
  line.str(input);
  return file;
}

void gxl::read_until(std::ifstream &file, std::string &input,
                     std::string &&to_find, Case u_l) {
  while (getline(file, input, u_l)) {
    if (input.starts_with(to_find)) {
      return;
    }
  }
  input = "end"; // 若没找到，就赋值为"end"代表结束了
  fmt::print("{} is not found in file.\n", to_find);
}

void gxl::read_until(std::ifstream &file, std::string &input, const std::vector<std::string> &to_find, Case u_l) {
  while (getline(file, input, u_l)) {
    for (auto &s: to_find) {
      if (input.starts_with(s)) {
        return;
      }
    }
  }
  input = "end"; // 若没找到，就赋值为"end"代表结束了
  for (auto &s: to_find) {
    fmt::print("{}\t", s);
  }
  fmt::print("are not found in file.\n");
}

std::string gxl::to_upper(std::string &str) {
  for (auto &ch: str) {
    ch = std::toupper(ch);
  }
  return str;
}

std::string gxl::read_str(FILE *file) {
  int value = 0;
  std::string str;
  while (true) {
    fread(&value, sizeof(int), 1, file);
    const char ch = static_cast<char>(value);
    if (ch == '\0') {
      break;
    }
    str += ch;
  }
  return str;
}

std::string gxl::to_upper(const std::string &str) {
  std::string str_upper{str};
  for (auto &ch: str_upper) {
    ch = std::toupper(ch);
  }
  return str_upper;
}

void gxl::write_str(const char *str, FILE *file) {
  int value = 0;
  while (*str != '\0') {
    value = static_cast<int>(*str);
    fwrite(&value, sizeof(int), 1, file);
    ++str;
  }
  constexpr char null_char = '\0';
  value = static_cast<int>(null_char);
  fwrite(&value, sizeof(int), 1, file);
}

std::string gxl::read_str_MPI_ver(const MPI_File &file, MPI_Offset &offset, int n_bytes) {
  char *ch = new char[n_bytes + 1];
  MPI_File_read_at(file, offset, ch, n_bytes, MPI_CHAR, MPI_STATUS_IGNORE);
  ch[n_bytes] = '\0';
  offset += n_bytes;
  std::string str{ch};
  delete[]ch;
  return str;
}

std::string gxl::read_str_from_binary_MPI_ver(const MPI_File &file, MPI_Offset &offset) {
  int value = 0;
  std::string str;
  while (true) {
    MPI_File_read_at(file, offset, &value, 1, MPI_INT32_T, MPI_STATUS_IGNORE);
    offset += 4;
    const char ch = static_cast<char>(value);
    if (ch == '\0') {
      break;
    }
    str += ch;
  }
  return str;
}

void gxl::write_str_without_null(const char *str, const MPI_File &file, MPI_Offset &offset) {
  while (*str != '\0') {
    MPI_File_write_at(file, offset, str, 1, MPI_CHAR, MPI_STATUS_IGNORE);
    ++offset;
    ++str;
  }
}

void gxl::write_str(const char *str, const MPI_File &file, MPI_Offset &offset) {
  int value = 0;
  while (*str != '\0') {
    value = static_cast<int>(*str);
    MPI_File_write_at(file, offset, &value, 1, MPI_INT32_T, MPI_STATUS_IGNORE);
    offset += 4;
    ++str;
  }
  constexpr char null_char = '\0';
  value = static_cast<int>(null_char);
  MPI_File_write_at(file, offset, &value, 1, MPI_INT32_T, MPI_STATUS_IGNORE);
  offset += 4;
}

std::string gxl::replace(std::string &str, char from, char to) {
  std::replace(str.begin(), str.end(), from, to);
  return str;
}

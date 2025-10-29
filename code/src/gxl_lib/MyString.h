#pragma once
#include <string>
#include <sstream>
#include <mpi.h>
#include <vector>

namespace gxl {
// Some string functions
const std::string whiteSpace=" \n\t\r\f\v";

enum class Case { lower, upper, keep };

std::istream& getline(std::ifstream& file, std::string& input, Case u_l=Case::keep);

void to_stringstream(const std::string& str, std::istringstream& line);

std::istream& getline_to_stream(std::ifstream& file, std::string& input, std::istringstream& line, Case u_l=Case::keep);

void trim_left(std::string& string);

void read_until(std::ifstream& file, std::string& input, std::string&& to_find, Case u_l=Case::keep);
void read_until(std::ifstream& file, std::string& input, const std::vector<std::string> &to_find, Case u_l= Case::keep);

std::string to_upper(std::string& str);
std::string to_upper(const std::string& str);

void write_str(const char *str, FILE *file);
std::string read_str(FILE* file);
std::string read_str_from_binary_MPI_ver(const MPI_File &file, MPI_Offset &offset);
std::string read_str_MPI_ver(const MPI_File &file, MPI_Offset &offset, int n_bytes);

void write_str_without_null(const char *str, const MPI_File &file, MPI_Offset &offset);
void write_str(const char *str, const MPI_File &file, MPI_Offset &offset);

std::string replace(std::string &str, char from, char to);
}

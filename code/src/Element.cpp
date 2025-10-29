#include "Element.h"
#include <vector>
#include "Constants.h"
#include "fmt/format.h"

namespace cfd {
struct AtomicWeightData {
  std::string symbol;
  std::string fullname;
  double atomic_weight;
};

static const std::vector<AtomicWeightData> atomic_weight_table{
    {"H",  "hydrogen",      1.008},
    {"HE", "helium",        4.002602},
    {"LI", "lithium",       6.94},
    {"BE", "beryllium",     9.0121831},
    {"B",  "boron",         10.81},
    {"C",  "carbon",        12.011},
    {"N",  "nitrogen",      14.007},
    {"O",  "oxygen",        15.999},
    {"F",  "fluorine",      18.998403163},
    {"NE", "neon",          20.1797},
    {"NA", "sodium",        22.98976928},
    {"MG", "magnesium",     24.305},
    {"AL", "aluminum",      26.9815384},
    {"SI", "silicon",       28.085},
    {"P",  "phosphorus",    30.973761998},
    {"S",  "sulfur",        32.06},
    {"CL", "chlorine",      35.45},
    {"AR", "argon",         39.95},
    {"K",  "potassium",     39.0983},
    {"CA", "calcium",       40.078},
    {"SC", "scandium",      44.955908},
    {"TI", "titanium",      47.867},
    {"V",  "vanadium",      50.9415},
    {"CR", "chromium",      51.9961},
    {"MN", "manganese",     54.938043},
    {"FE", "iron",          55.845},
    {"CO", "cobalt",        58.933194},
    {"NI", "nickel",        58.6934},
    {"CU", "copper",        63.546},
    {"ZN", "zinc",          65.38},
    {"GA", "gallium",       69.723},
    {"GE", "germanium",     72.630},
    {"AS", "arsenic",       74.921595},
    {"SE", "selenium",      78.971},
    {"BR", "bromine",       79.904},
    {"KR", "krypton",       83.798},
    {"RB", "rubidium",      85.4678},
    {"SR", "strontium",     87.62},
    {"Y",  "yttrium",       88.90584},
    {"ZR", "zirconium",     91.224},
    {"NB", "nobelium",      92.90637},
    {"MO", "molybdenum",    95.95},
    {"TC", "technetium",    -1.0},
    {"RU", "ruthenium",     101.07},
    {"RH", "rhodium",       102.90549},
    {"PD", "palladium",     106.42},
    {"Ag", "silver",        107.8682},
    {"CD", "cadmium",       112.414},
    {"IN", "indium",        114.818},
    {"SN", "tin",           118.710},
    {"SB", "antimony",      121.760},
    {"TE", "tellurium",     127.60},
    {"I",  "iodine",        126.90447},
    {"XE", "xenon",         131.293},
    {"CS", "cesium",        132.90545196},
    {"BA", "barium",        137.327},
    {"LA", "lanthanum",     138.90547},
    {"CE", "cerium",        140.116},
    {"PR", "praseodymium",  140.90766},
    {"ND", "neodymium",     144.242},
    {"PM", "promethium",    -1.0},
    {"SM", "samarium",      150.36},
    {"EU", "europium",      151.964},
    {"GD", "gadolinium",    157.25},
    {"TB", "terbium",       158.925354},
    {"DY", "dysprosium",    162.500},
    {"HO", "holmium",       164.930328},
    {"ER", "erbium",        167.259},
    {"TM", "thulium",       168.934218},
    {"YB", "ytterbium",     173.045},
    {"LU", "lutetium",      174.9668},
    {"Hf", "hafnium",       178.49},
    {"TA", "tantalum",      180.94788},
    {"W",  "tungsten",      183.84},
    {"RE", "rhenium",       186.207},
    {"OS", "osmium",        190.23},
    {"IR", "iridium",       192.217},
    {"PT", "platinum",      195.084},
    {"AU", "gold",          196.966570},
    {"HG", "mercury",       200.592},
    {"TL", "thallium",      204.38},
    {"PB", "lead",          207.2},
    {"BI", "bismuth",       208.98040},
    {"PO", "polonium",      -1.0},
    {"AT", "astatine",      -1.0},
    {"RN", "radon",         -1.0},
    {"FR", "francium",      -1.0},
    {"RA", "radium",        -1.0},
    {"AC", "actinium",      -1.0},
    {"TH", "thorium",       232.0377},
    {"PA", "protactinium",  231.03588},
    {"U",  "uranium",       238.02891},
    {"NP", "neptunium",     -1.0},
    {"PU", "plutonium",     -1.0},
    {"AM", "americium",     -1.0},
    {"CM", "curium",        -1.0},
    {"BK", "berkelium",     -1.0},
    {"CF", "californium",   -1.0},
    {"ES", "einstiunium",   -1.0},
    {"FM", "fermium",       -1.0},
    {"MD", "mendelevium",   -1.0},
    {"NO", "nobelium",      -1.0},
    {"LR", "lawrencium",    -1.0},
    {"RF", "rutherfordium", -1.0},
    {"DB", "dubnium",       -1.0},
    {"SG", "seaborgium",    -1.0},
    {"BH", "bohrium",       -1.0},
    {"HS", "hassium",       -1.0},
    {"MT", "meitnerium",    -1.0},
    {"DS", "darmstadtium",  -1.0},
    {"RG", "roentgenium",   -1.0},
    {"CN", "copernicium",   -1.0},
    {"NH", "nihonium",      -1.0},
    {"GL", "flerovium",     -1.0},
    {"MC", "moscovium",     -1.0},
    {"LV", "livermorium",   -1.0},
    {"TS", "tennessine",    -1.0},
    {"OG", "oganesson",     -1.0},
};

struct isotope_weight_data {
  std::string symbol;
  std::string full_name;
  double atomic_weight;
  int atomic_number;
};

static const std::vector<isotope_weight_data> isotope_weight_table = {
    {"D",  "deuterium", 2.0141017781,             1},
    {"Tr", "tritium",   3.0160492820,             1},
    {"E",  "electron",  electron_mass * avogadro, 0},
};


double Element::get_atom_weight() const {
  double atom_weight{0};
  for (auto &it: atomic_weight_table) {
    if (name == it.symbol) {
      atom_weight = it.atomic_weight;
      break;
    }
  }
  if (atom_weight > 0) {
    return atom_weight;
  }
  for (auto &it: isotope_weight_table) {
    if (name == it.symbol) {
      return it.atomic_weight;
    }
  }
  fmt::print("No such element. Please check if the input is reasonable.\n");
  return -1.0;
}
}

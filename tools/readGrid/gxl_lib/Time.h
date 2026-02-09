#pragma once

namespace gxl{
class Time {
  double start_time{0};
  double last_record_time{0};
public:
  double elapsed_time{0};
  double step_time{0};

  Time();

  double record_current();

  double get_elapsed_time();

};
}

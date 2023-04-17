// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

namespace dr {

#if DR_FORMAT

class logger {
public:
  void set_file(std::ofstream &fout) { fout_ = &fout; }

  template <typename... Args>
  void debug(const nostd::source_location &location,
             fmt::format_string<Args...> format, Args &&...args) {
    if (fout_) {
      *fout_ << location.file_name() << ":" << location.line() << ": "
             << fmt::format(format, std::forward<Args>(args)...);
      fout_->flush();
    }
  }
  template <typename... Args>
  void debug(fmt::format_string<Args...> format, Args &&...args) {
    if (fout_) {
      *fout_ << fmt::format(format, std::forward<Args>(args)...);
      fout_->flush();
    }
  }

private:
  std::ofstream *fout_ = nullptr;
};

#else

class logger {
public:
  void set_file(std::ofstream &fout) { fout_ = &fout; }

  template <typename... Args>
  void debug(const nostd::source_location &location, std::string format,
             Args &&...args) {}
  template <typename... Args> void debug(std::string format, Args &&...args) {}

private:
  std::ofstream *fout_ = nullptr;
};

#endif

inline logger drlog;

} // namespace dr

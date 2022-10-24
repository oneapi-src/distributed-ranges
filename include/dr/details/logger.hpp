namespace lib {

class logger {
public:
  void set_file(std::ofstream &fout) { fout_ = &fout; }

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

inline logger drlog;

} // namespace lib

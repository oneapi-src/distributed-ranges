template <typename T> inline MPI_Datatype mpi_data_type() {
  assert(false);
  return MPI_CHAR;
}
template <> inline MPI_Datatype mpi_data_type<int>() { return MPI_INT; }
template <> inline MPI_Datatype mpi_data_type<const int>() { return MPI_INT; }
template <> inline MPI_Datatype mpi_data_type<float>() { return MPI_FLOAT; }
template <> inline MPI_Datatype mpi_data_type<const float>() {
  return MPI_FLOAT;
}
template <> inline MPI_Datatype mpi_data_type<double>() { return MPI_DOUBLE; }
template <> inline MPI_Datatype mpi_data_type<const double>() {
  return MPI_DOUBLE;
}

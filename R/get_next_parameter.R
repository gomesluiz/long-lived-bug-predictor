get_next_parameter <- function(.metrics_file, .parameters) {
  last_row <- nrow(.metrics_file)

  if (last_row == 0) {
    return(1)
  }
  if (last_row >= nrow(.parameters)) {
    return(-1)
  }

  return(last_row + 1)
}

get_metrics_file <- function(folder=".", suffix){
  # Gets the last evaluation file processed.
  #
  # Args:
  #   folder: place where evaluation files are stored.
  #
  # Returns:
  #   The full name of the last evalution file.
  current.folder = getwd()
  setwd(folder)
  file_pattern <- paste("\\_", suffix, sep="")
  files <- list.files(pattern = file_pattern)
  files <- files[sort.list(files, decreasing = TRUE)]
  setwd(current.folder)
  
  if (is.na(files[1])) return(NA)
  
  return(file.path(folder, files[1]))
}

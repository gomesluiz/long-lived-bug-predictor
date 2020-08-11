format_file_name <- function(.path=".", .timestamp, .suffix) {
  #' Format evaluation files name.
  #'
  #' @param .path The place where evaluation files are stored.
  #' @param .timestamp
  #' @param .suffix 
  #'  
  #' @return The file name formatted.
  #'
  name  <- sprintf("%s_%s", .timestamp, .suffix)
  name  <- file.path(.path, name)
  return(name)
}
insert_one_evaluation_data <- function(evaluation.file, one.evaluation) {
  # Insert one evaluation data metrics
  #
  # Args:
  #   evaluation.file: name of evaluation file.
  #   one.evaluation : evaluation dataframe.
  #
  # Returns:
  #   nothing.
  #
  
  if (file.exists(evaluation.file)){
    all.evaluations <- read_csv(evaluation.file)
    if (nrow(all.evaluations) > 0) {
       all.evaluations <- rbind(all.evaluations , one.evaluation)
    } else {
      all.evaluations <- one.evaluation
    }
  } else {
    all.evaluations <- one.evaluation
  }
  
  write_csv( all.evaluations , evaluation.file)
}
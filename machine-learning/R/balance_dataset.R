# balancing methods
UNBALANCED    <- "unbalanced"
DOWNSAMPLE    <- "downsample"
SMOTEMETHOD   <- "smote"
MANUALMETHOD  <- "manual"

balance_dataset <- function(.x, .label, .out, .fnc) {
  #
  #
  # Args:
  #
  #
  # Returns:
  #
  keep <- !(names(.x) %in% .out)
  
  if (.fnc == UNBALANCED) {
    flog.trace("[balance_dataset]: balancing method: %s", UNBALANCED)
    result <- .x[, keep]
  } else if (.fnc == MANUALMETHOD) {
    flog.trace("[balance_dataset]: balancing method: %s", MANUALMETHOD)
    result <- do_down_sampling(.x[, keep], .label)
  } else if (.fnc == DOWNSAMPLE) {
    flog.trace("[balance_dataset]: balancing method: %s", DOWNSAMPLE)
    keep <- !(names(.x) %in% c(.out, .label))
    result <- downSample(.x[, keep], y = .x[, .label], yname = .label)
  } else if (.fnc == SMOTEMETHOD) {
    flog.trace("[balance_dataset]: balancing method: %s", SMOTEMETHOD)
    keep <- !(names(.x) %in% c(.out, .label))
    result <- SMOTE(.x[, keep], .x[, .label], K = 3, dup_size = 0)$data
    colnames(result)[colnames(result) == "class"] <- .label
    result[, .label] = as.factor(result[, .label])
  } else {
    stop("balance_dataset: balanced function unknown!")
  }
  return(result)
}
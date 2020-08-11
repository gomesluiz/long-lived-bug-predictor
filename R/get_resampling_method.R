library(futile.logger)

resampling_methods = list(
  none  = "none",
  boot  = "boot",
  cv10  = "cv10",
  loocv = "LOOCV",
  lgocv = "LGOCV",
  repeatedcv = "repeatedcv"
)

get_resampling_method <- function(.method) {
  #' Returns the caret resampling method control.
  #'
  #' @param method The method of resampling.
  #'
  #' @return The caret resampling method control.
  #' 
  if (.method == resampling_methods[["cv10"]]) {
    result <- caret::trainControl(method = "cv", number = 10, search = "grid", savePredictions = 'final')
    flog.trace("[get_resampling_method] Resampling model %s", "10cv") 
  } else if (.method == resampling_methods[["repeatedcv"]]) {
    result <- caret::trainControl(method = .method, number = 5, repeats = 2, search = "grid", savePredictions = 'final')
    flog.trace("[get_resampling_method] Resampling model %s", "repeatedcv") 
  } else if (.method %in% c(resampling_methods[["none"]]
                            , resampling_methods[["boot"]]
                            , resampling_methods[["loocv"]]
                            , resampling_methods[["lgocv"]])) {
    flog.trace("[get_resampling_method] Resampling model %s", .method) 
    result <- caret::trainControl(method = .method, search = "grid", savePredictions = 'final')
  } else {
    stop("Invalid resampling method!", call. = FALSE)  
  }
  return (result)
}
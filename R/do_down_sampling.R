do_down_sampling <- function(x, class) {
  #' Applies down sampling balacing method over a dataset.
  #'
  #' @param x       The unbalanced dataset
  #' @param class   The class attribute
  #'
  #' @return A balanced dataset.
  #'
  set.seed(1234)
  
  class.0 <- x[which(x[, class] == 0), ]
  class.1 <- x[which(x[, class] == 1), ]
  
  size = min(nrow(class.0), nrow(class.1))
  
  sample.class.0 <- sample_n(class.0, size, replace = FALSE)
  sample.class.1 <- sample_n(class.1, size, replace = FALSE)
  
  result <- rbind(sample.class.0, sample.class.1)
  
  return(result)
}

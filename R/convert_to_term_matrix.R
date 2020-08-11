#' @description 
#' Convert a report dataset to document term matrix using the feature parameter.
#' 
#' @author 
#' Luiz Alberto (gomes.luiz@gmail.com)
#' 
#' @param .dataset a bug report dataset
#' @param .feature a textual feature
#' @param .nterms  a number of matrix terms 
#' 
convert_to_term_matrix <- function(.dataset, .feature, .nterms){
  source <- .dataset[, c('bug_id', .feature)]
  corpus <- clean_corpus(source)
  dtm    <- tidy(make_dtm(corpus, .nterms))
  names(dtm)      <- c("bug_id", "term", "count")
  term.matrix     <- dtm %>% spread(key = term, value = count, fill = 0)
  result.dataset  <- merge(
    x = .dataset[, c('bug_id', 'bug_fix_time')],
    y = term.matrix,
    by.x = 'bug_id',
    by.y = 'bug_id'
  )
  # replace the SMOTE reserved words.
  colnames(result.dataset)[colnames(result.dataset) == "class"]  <- "Class"
  colnames(result.dataset)[colnames(result.dataset) == "target"] <- "Target"
  
  return (result.dataset)
}
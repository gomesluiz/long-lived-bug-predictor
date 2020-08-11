make_dtm <- function(corpus, n = 100){
  # makes a document term matrix from a corpus.
  #
  #
  # Args:
  #   corpus: The document corpus.
  #   n     : The number of terms of document matrix.
  #
  # Returns:
  #   The document matrix from corpus with n terms.
  dtm           <- DocumentTermMatrix(corpus, control = list(weighting = weightTfIdf))
  
  freq <- colSums(as.matrix(dtm))
  freq <- order(freq, decreasing = TRUE)
  
  return(dtm[, freq[1:n]])
}
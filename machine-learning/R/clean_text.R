clean_text <- function(text){
  # Clean a text replacing abbreviations, contractions and symbols. Furthermore,
  # this function, converts the text to lower case.
  #
  # Args:
  #   text: text to be cleanned.
  #
  # Returns:
  #   The text cleanned.
  text <- replace_abbreviation(text, ignore.case = TRUE)
  text <- replace_contraction(text, ignore.case = TRUE)
  text <- replace_symbol(text)
  text <- tolower(text)
  return(text)
}
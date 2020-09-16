library(readr)
library(caret)
set.seed(2358)
print("Pre-processing")
clinico  <- read_delim("test/clinico.csv", ";", escape_double = FALSE, trim_ws = TRUE)
clinico  <- as.data.frame(clinico)
clinico$Diagnostico <- as.factor(ifelse(clinico$Diagnostico=="Saudavel", 0, 1))
in_train <- createDataPartition(y=clinico$Diagnostico, p=.75, list=FALSE)

features <- c('Idade', 'Sexo', 'Peso', 'Manchas', 'Temp', 'Internacoes')
target   <- c('Diagnostico')

X_train  <- clinico[in_train, features]
y_train  <- clinico[in_train, target]

X_test   <- clinico[-in_train, features]
y_test   <- clinico[-in_train, target]

print("Training model")
model <- train(
  x = X_train,
  y = y_train,
  method    = "rf",
  trControl = trainControl(method = "repeatedcv", number = 2, repeats = 2, search = "grid", savePredictions = "final"),
  tuneGrid    = expand.grid(mtry=c(2)),
  metric      = "Accuracy"
)
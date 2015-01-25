library(caret)
library(Amelia)
training <- read.csv('pml-training.csv', na.strings=c("NA","#DIV/0!",""))
test <- read.csv('pml-testing.csv', na.strings=c("NA","#DIV/0!",""))

#Classe is the target variable, let's take a look at it
#According to the dataset documentation>
# A - Exercise done according to specifications
# B - Throwing the elbows to the front
# C - Lifting the dumbbell only halfway
# D - Lowering the dumbbell only halfway
table(training$classe)


############### FEATURE SELECTION
############### Make the same transformation in the testing and the test set
#Get and remove near zero covariates from the data
nzv <- nearZeroVar(training, saveMetrics=TRUE)
training <- training[,-which(nzv$nzv==TRUE)]
test <- test[,-which(nzv$nzv==TRUE)]

### Let's take a look for the missing values
#Draw a missing map with red for missing values, lightgreen for observed values
#and darkred for missing values
missmap(training, col=c("darkred","lightgreen"), legend=FALSE, rank.order=FALSE)

#The previous plot shows that there are a lot of almost completely
#empty (and therefore useless) columns in the dataset, so let's remove them

#Get a vector with the total missing values for each column
totalMissingValues = sapply(training, function(x) {sum(is.na(x))})
hist(totalMissingValues)

#Here we can see the useless columns have more than 15,000 missing values, 
#so that's will be our simple rule to drop columns

#Get the names of the columns to remove
columnsToRemove = names(totalMissingValues[totalMissingValues>15000])
training <- training[,!names(training) %in% columnsToRemove]
test <- test[,!names(test) %in% columnsToRemove]


# According to the documentation, the first seven variables are related to the subject
# and some other descriptive values, so we can also leave them out
training <- training[,-c(1:7)]
test <- test[,-c(1:7)]

################### Train control
# The method for train control will be 10 fold Cross Validation
train_control <- trainControl(method="cv", number=10, allowParallel = TRUE)
set.seed(1234)

#Build the predictive models
#Let's build two different models: 
#Trees and random forest

############### Model 1: Trees
startTime <- Sys.time()
model1 <- train(classe ~ ., data = training, method = "rpart", trControl = train_control)
endTime <- Sys.time()
model1Time <- endTime - startTime
print(model1Time)
round(max(head(model1$results)$Accuracy), 3)

############### Model 2: Random Forest
startTime <- Sys.time()
model2 <- train(classe ~ ., data = training, method = "rf", trControl = train_control)
endTime <- Sys.time()
model2Time <- endTime - startTime
print(model2Time)
round(max(head(model2$results)$Accuracy), 3)

# Random fortest yields much better results, so let's make our predictions based
# on this model

############ Make the predictions
predictions2 <- predict(model1, test)

#Make the files to submit to the class
#Code taken from coursera assignment specification
pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}

pml_write_files(predictions2)





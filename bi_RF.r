#packages
library(randomForest)
library(tidyverse)
library(skimr)
library(DataExplorer)
library(caret)
library(pROC)

#training dataset&testing dataset
traindata <- read.csv('',header=T,sep=",")
testdata <- read.csv('',header=T,sep=",")

#make the formula of selected features
form_cls <-as.formula(
  paste0('BCVA_5y_WHO ~ ',paste(colnames(traindata)[c(#selected features)],collapse = '+'))
)
form_cls

#build the model
control <- trainControl(method='repeatedcv', 
                        number=10, 
                        repeats=3,
                        search='grid')
tunegrid <- expand.grid(.mtry = (1:10)) #the number of selected features
modellist <- list()
rf_default <- train(form_cls, 
                    data=traindata, 
                    method='rf', 
                    tuneGrid=tunegrid, 
                    trControl=control)

print(rf_default) #find the mtry

set.seed(42) 
rf_train <- randomForest(
  form_cls,
  data = traindata,
  ntree =1000, 
  mtry = 4, 
  importance = T
)
plot(rf_train)#find the ntree

#build the model with the best mtry and ntree
set.seed(42) 
fit_rf_clsm <- randomForest(
  form_cls,
  data = traindata,
  ntree =200, 
  mtry = 4, 
  importance = T 
)
fit_rf_clsm

#the outcome of training dataset
trainpredprob <- predict(fit_rf_clsm, newdata = traindata, type = 'prob')
trainroc <- roc(response = traindata$BCVA_5y_WHO, predictor = trainpredprob[,2])

#get the thresholds
bestp <- trainroc$thresholds[
  which.max(trainroc$sensitivities + trainroc$specificities - 1)
]
bestp

#the confusionmatrix of training dataset
trainpredlab <- as.factor(ifelse(trainpredprob[,2] > bestp,1,0)) 
confusionMatrix(data = trainpredlab,
                reference = traindata$BCVA_5y_WHO,
                positive = '1',
                mode = 'everything')

#the outcome of testing dataset
testpredprob <- predict(fit_rf_clsm, newdata = testdata, type = 'prob')
testpredlab <- as.factor(ifelse(testpredprob[,2] > bestp, 1, 0)) 
confusionMatrix(data = testpredlab,
                reference = testdata$BCVA_5y_WHO,
                positive = '1',
                mode = 'everything')

#feature importance
importance(fit_rf_clsm)
varImpPlot(fit_rf_clsm,main = 'Variable Importance Plot')


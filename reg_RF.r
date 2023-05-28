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

#just BCVA at 3y as an example
#make the formula of selected features
form_reg <-as.formula(
  paste0('BCVA_3y ~ ',paste(colnames(traindata)[c(#selected features)],collapse = '+'))
)

#build the model
control <- trainControl(method='repeatedcv', 
                        number=10, 
                        repeats=3,
                        search='grid')
tunegrid <- expand.grid(.mtry = (1:9)) #the number of selected features
modellist <- list()
rf_default <- train(form_reg, 
                    data=traindata, 
                    method='rf', 
                    tuneGrid=tunegrid, 
                    trControl=control)

print(rf_default) #find the mtry

set.seed(100)
rf_train<-randomForest(
  form_reg,
  data = traindata,
  ntree =1000, 
  mtry = 4, 
  importance = T 
) 
plot(rf_train) #find the ntree

#build the model with the best mtry and ntree
set.seed(42) 
fit_rf_reg <- randomForest(
  form_reg,
  data = traindata,
  ntree =400, 
  mtry = 4, 
  importance = T 
)
fit_rf_reg

#the outcome of training dataset
trainpred <- predict(fit_rf_reg, newdata = traindata)
defaultSummary(data.frame(obs = traindata$BCVA_3y, pred = trainpred))

#the outcome of testing dataset
testpred <- predict(fit_rf_reg, newdata = testdata)
defaultSummary(data.frame(obs = testdata$BCVA_3y, pred = testpred))

#feature importance
importance(fit_rf_reg)
varImpPlot(fit_rf_reg,main = 'Variable Importance Plot')

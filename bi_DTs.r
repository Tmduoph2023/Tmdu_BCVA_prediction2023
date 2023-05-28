#packages
library(rpart)
library(rpart.plot)
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
set.seed(42) 
fit_dt_clsm <- rpart(
  form_cls,
  data = traindata,
  method = 'class', #classification model
  parms = list(split = 'gini'), 
  control = rpart.control(cp = 0.01) 
)
#classification tree
printcp(fit_dt_clsm)
plotcp(fit_dt_clsm) #find the size of tree

# Use cross validation
train_control <- trainControl(method = "repeatedcv",  
                              number = 10,             
                              repeats = 3)
tune_grid = expand.grid(cp=c(0.031))
fit_dt_clsm_pruned <- train(form_cls,
                        data=traindata,                 
                        method="rpart",                     
                        trControl= train_control,           
                        tuneGrid = tune_grid,              
                        maxdepth = 5,                       
                        minbucket=5)
fit_dt_clsm_pruned

#the outcome of training dataset
trainpredprob <- predict(fit_dt_clsm_pruned, newdata = traindata, type = 'prob')
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
testpredprob <- predict(fit_dt_clsm_pruned, newdata = testdata, type = 'prob')
testpredlab <- as.factor(ifelse(testpredprob[,2] > bestp, 1, 0)) 
confusionMatrix(data = testpredlab,
                reference = testdata$BCVA_5y_WHO,
                positive = '1',
                mode = 'everything')


#show the feature importanve
fit_dt <- prune(fit_dt_clsm, cp=0.031)
fit_dt$variable.importance

#show the tree
prp(fit_dt,
    type = 2,
    extra = 104,
    tweak = 1,
    fallen.leaves = TRUE,
    main = 'Decision Tree')

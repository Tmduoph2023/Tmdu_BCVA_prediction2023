#packages
library(xgboost)
library(tidyverse)
library(skimr)
library(DataExplorer)
library(caret)
library(pROC)

#training dataset&testing dataset
traindata <- read.csv('',header=T,sep=",")
testdata <- read.csv('',header=T,sep=",")

#get the valisation set
set.seed(42)
trains <- createDataPartition(
  y=traindata$BCVA_5y_WHO,
  p=0.9,
  list = FALSE)
data_valid <- traindata[-trains,]
data_train <- traindata[trains,]


#make the formula of selected features
form_cls <-as.formula(
  paste0('BCVA_5y_WHO ~ ',paste(colnames(traindata)[c(#selected features)],collapse = '+'))
)
form_cls

#One-Hot Encoding
dvfunc <- dummyVars(~., data = data_train[,c(#selected features)], fullRank = TRUE)
data_trainx <- predict(dvfunc, newdata = data_train[,c(#selected features)])
data_trainy <- ifelse(data_train$BCVA_5y_WHO == 0,0,1)

data_validx <- predict(dvfunc, newdata = data_valid[,c(#selected features)])
data_validy <- ifelse(data_valid$BCVA_5y_WHO == 0,0,1)

data_testx <- predict(dvfunc, newdata = data_test[,c(#selected features)])
data_testy <- ifelse(data_test$BCVA_5y_WHO == 0,0,1)

dtrain <- xgb.DMatrix(data = data_trainx,
                      label = data_trainy)

dvalid <- xgb.DMatrix(data = data_validx,
                      label = data_validy)

dtest <- xgb.DMatrix(data = data_testx,
                     label = data_testy)

watchlist <- list(train = dtrain, test = dvalid)

#get the best hyperparameter by grid search
xgb_caret <- train(x = as.matrix(data_train[,c(#selected features)]),
                   y = data_train$BCVA_5y_WHO,
                   method = 'xgbTree',
                   objective = 'binary:logistic',
                   trControl = trainControl(method = 'repeatedcv',
                                            number = 10,
                                            repeats = 3,
                                            verboseIter = TRUE),
                   tuneGrid = expand.grid(nrounds = c(500,1000,1500),
                                          eta = c(0.01,0.05,0.1,0.2),
                                          max_depth = c(2,4,6),
                                          colsample_bytree = c(0.5,1),
                                          subsample = c(0.5,1),
                                          gamma = c(0,0.1,0.2,0.3,0.4),
                                          min_child_weight = c(0,2,4,6)))

names(xgb_caret)
xgb_caret$bestTune
xgb_caret$method
summary(xgb_caret)

#build the model
fit_xgb_cls <- xgb.train(
  data = dtrain,
  
  eta = 0.01,
  gamma = 0.3,
  max_depth = 2,
  subsample = 0.5,
  colsample_bytree = 1,
  
  objective = 'binary:logistic',
  
  nrounds = 500,
  watchlist = watchlist,
  verbose = 1,
  print_every_n = 100,
  early_stopping_rounds = 200
)

#the outcome of training dataset
trainpredprob <- predict(fit_xgb_cls, 
                         newdata = dtrain)
trainroc <- roc(response = data_train$BCVA_5y_WHO, 
                predictor = trainpredprob)

#get the thresholds
bestp <- trainroc$thresholds[
  which.max(trainroc$sensitivities + trainroc$specificities - 1)
]
bestp

#the confusionmatrix of training dataset
trainpredlab <- as.factor(ifelse(trainpredprob> bestp,1,0)) 
confusionMatrix(data = trainpredlab,
                reference = data_train$BCVA_5y_WHO,
                positive = '1',
                mode = 'everything')

#the outcome of testing dataset
testpredprob <- predict(fit_xgb_cls,newdata = dtest)
testpredlab <- as.factor(ifelse(testpredprob> bestp,1,0)) 
confusionMatrix(data = testpredlab,
                reference = data_test$BCVA_5y_WHO,
                positive = '1',
                mode = 'everything')

#feature importance
importance_matrix <- xgb.importance(model = fit_xgb_cls)
print(importance_matrix)
xgb.plot.importance(importance_matrix = importance_matrix,
                    measure = 'Cover')

#SHAP
xgb.plot.shap(data = data_trainx,
              model = fit_xgb_cls,
              top_n = 5)



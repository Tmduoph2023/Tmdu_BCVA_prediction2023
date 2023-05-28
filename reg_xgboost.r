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

#just BCVA at 3y as an example
#get the valisation set
set.seed(42)
trains <- createDataPartition(
  y=traindata$BCVA_3y,
  p=0.9,
  list = FALSE)
data_valid <- traindata[-trains,]
data_train <- traindata[trains,]

#One-Hot Encoding
dvfunc <- dummyVars(~., data = data_train[,c(#selected features)], fullRank = TRUE)
data_trainx <- predict(dvfunc, newdata = data_train[,c(#selected features)])
data_trainy <- data_train$BCVA_3y

data_validx <- predict(dvfunc, newdata = data_valid[,cc(#selected features)])
data_validy <- data_valid$BCVA_3y

data_testx <- predict(dvfunc, newdata = data_test[,cc(#selected features)])
data_testy <- data_test$BCVA_3y

dtrain <- xgb.DMatrix(data = data_trainx,
                      label = data_trainy)

dvalid <- xgb.DMatrix(data = data_validx,
                      label = data_validy)

dtest <- xgb.DMatrix(data = data_testx,
                     label = data_testy)

watchlist <- list(train = dtrain, test = dvalid)

#get the best hyperparameter by grid search
xgb_caret <- train(x = as.matrix(data_train[,c(#selected features)]),
                   y = data_train$BCVA_3y,
                   method = 'xgbTree',
                   objective = 'reg:squarederror',
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
fit_xgb_reg <- xgb.train(
  data = dtrain,
  eta = 0.01,
  gamma = 0.3,
  max_depth = 6,
  subsample = 0.5,
  colsample_bytree = 1,
  objective = 'reg:squarederror',
  nrounds = 500,
  watchlist = watchlist,
  verbose = 1,
  print_every_n = 100,
  min_child_weight = 6,  
  early_stopping_rounds = 200
)

#the outcome of training dataset
trainpred <- predict(fit_xgb_reg, newdata = dtrain)
defaultSummary(data.frame(obs = data_train$BCVA_3y, 
                          pred = trainpred))

#the outcome of testing dataset
testpred <- predict(fit_xgb_reg, newdata = dtest)
defaultSummary(data.frame(obs = data_test$BCVA_3y, 
                          pred = testpred))

#feature importance
importance_matrix <- xgb.importance(model = fit_xgb_reg)
print(importance_matrix)
xgb.plot.importance(importance_matrix = importance_matrix,
                    measure = 'Cover')

#SHAP
xgb.plot.shap(data = data_trainx,
              model = fit_xgb_reg,
              top_n = 5)



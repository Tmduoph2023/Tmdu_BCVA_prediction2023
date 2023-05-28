#packages
library(e1071)
library(tidyverse)
library(skimr)
library(caret)
library(pROC)

#training dataset&testing dataset
traindata <- read.csv('',header=T,sep=",")
testdata <- read.csv('',header=T,sep=",")

#make the formula of selected features
form_cls <-as.formula(
  paste0('BCVA_5y_WHO ~ ',paste(colnames(traindata)[c(#selected features)],collapse = '+'))
  )

#build the model and try different kernels and find the best one
set.seed(123)
linear.tune <- tune.svm(form_cls, data = traindata, 
                        kernel = "linear", 
                        cost = c(0.001, 0.01, 0.1, 1, 5, 10),
                        probability = T)
poly.tune <- tune.svm(form_cls, data = traindata, 
                      kernel = "polynomial", 
                      degree = c(3, 4, 5), 
                      coef0 = c(0.1, 0.5, 1, 2, 3, 4),
                      probability = T)
rbf.tune <- tune.svm(form_cls, data = traindata, 
                     kernel = "radial", 
                     gamma = c(0.1, 0.5, 1, 2, 3, 4),
                     probability = T)
sigmoid.tune <- tune.svm(form_cls, data = traindata, 
                         kernel = "sigmoid", 
                         gamma = c(0.1, 0.5, 1, 2, 3, 4),
                         coef0 = c(0.1, 0.5, 1, 2, 3, 4),
                         probability = T)
summary(linear.tune) # use linear.tuneas an example
final_model <- linear.tune$best.model

#the outcome of training dataset
trainpred <- predict(final_model, newdata = traindata, probability = T)
trainpredprob <- attr(trainpred, 'probabilities')
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
testpred <- predict(final_model, newdata = testdata, probability = T)
testpredprob <- attr(testpred, 'probabilities')
testpredlab <- as.factor(ifelse(testpredprob[,2] > bestp,1,0))
confusionMatrix(data = testpredlab,
                reference = testdata$BCVA_5y_WHO,
                positive = '1',
                mode = 'everything')






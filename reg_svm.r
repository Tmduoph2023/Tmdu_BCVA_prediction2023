#packages
library(e1071)
library(tidyverse)
library(skimr)
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

#set the cross-validation
set.seed(123)
tc <- tune.control(sampling = "cross",
                   cross = 10,
                   nrepeat =3)

#try different kernels on traning dataset
set.seed(123)
linear.tune <- tune.svm(form_reg, data = traindata, 
                        kernel = "linear", 
                        cost = c(0.001,0.01,0.1,0.25,0.5,0.75,1,1.25,5,10),
                        tunecontrol = tc)
summary(linear.tune)
best.linear <- linear.tune$best.model
#the outcome of testing dataset
testpred.linear <- predict(best.linear, newdata = testdata)
defaultSummary(data.frame(obs = testdata$BCVA_3y, pred = testpred.linear))
plot(linear.tune)

#try different kernels on traning dataset
set.seed(123)
poly.tune <- tune.svm(form_reg, data = traindata,
                      kernel = "polynomial", 
                      degree = c(3, 4, 5), 
                      coef0 = c(0.1, 0.5, 1, 2, 3, 4),
                      tunecontrol = tc)
summary(poly.tune)
best.poly <- poly.tune$best.model
#the outcome of testing dataset
testpred.poly <- predict(best.poly, newdata = testdata)
defaultSummary(data.frame(obs = testdata$BCVA_3y, pred = testpred.poly))

#try different kernels on traning dataset
set.seed(123)
rbf.tune <- tune.svm(form_reg, data = traindata, 
                     kernel = "radial", 
                     gamma = c(0.1, 0.5, 1, 2, 3, 4),
                     tunecontrol = tc)
summary(rbf.tune)
best.rbf <- rbf.tune$best.model
#the outcome of testing dataset
testpred.rbf <- predict(best.rbf, newdata = testdata)
defaultSummary(data.frame(obs = testdata$BCVA_3y, pred = testpred.rbf))

#try different kernels on traning dataset
set.seed(123)
sigmoid.tune <- tune.svm(form_reg, data = traindata, 
                         kernel = "sigmoid", 
                         gamma = c(0.1, 0.5, 1, 2, 3, 4),
                         coef0 = c(0.1, 0.5, 1, 2, 3, 4),
                         tunecontrol = tc)
summary(sigmoid.tune)
best.sigmoid <- sigmoid.tune$best.model
#the outcome of testing dataset
testpred.sigmoid <- predict(best.sigmoid, newdata = testdata)
defaultSummary(data.frame(obs = testdata$BCVA_3y, pred = testpred.sigmoid))

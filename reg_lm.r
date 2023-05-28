#packages
library(e1071)
library(tidyverse)
library(skimr)
library(caret)
library(pROC)
library(MLmetrics)

#training dataset&testing dataset
traindata
testdata

#make the formula of choosed features
form_reg3 <-as.formula(
  paste0('BCVA_3y ~ ',paste(colnames(traindata)[c(#choosed features)],collapse = '+'))
)
form_reg5 <-as.formula(
  paste0('BCVA_5y ~ ',paste(colnames(traindata)[c(#choosed features)],collapse = '+'))
)

#build the model
set.seed(123)
train.control <- trainControl(method= "repeatedcv", 
                            number=10,
                            repeats=3)
modelCV3 <- train(form_reg3,
               data=traindata,
               method= "lm",  
               trControl=train.control)
modelCV5 <- train(form_reg5,
               data=traindata,
               method= "lm",  
               trControl=train.control)

modelCV3 <- modelCV3$finalModel
modelCV5 <- modelCV5$finalModel

#the outcome of training dataset
trainpred3 <- predict(modelCV3, newdata = traindata)
defaultSummary(data.frame(obs = traindata$BCVA_3y, pred = trainpred3))
trainpred5 <- predict(modelCV5, newdata = traindata)
defaultSummary(data.frame(obs = traindata$BCVA_5y, pred = trainpred5))

#the outcome of testing dataset
testpred3 <- predict(modelCV3, newdata = testdata)
defaultSummary(data.frame(obs = testdata$BCVA_3y, pred = testpred3))
testpred5 <- predict(modelCV5, newdata = testdata)
defaultSummary(data.frame(obs = testdata$BCVA_5y, pred = testpred5))

#the calibration belt
Observed3 = data.frame(Observed3=testdata$BCVA_3y)
Predicted3 = data.frame(Predicted3 = predict(modelCV3, newdata = testdata))
comparison3 = as.data.frame(cbind(Observed3,Predicted3))
ggplot(aes(x=Predicted3,y=Observed3),data=comparison3)+ggtitle("Calibration")+
  geom_smooth()+
  geom_segment(aes(x=0,y=0,xend=1.6,yend=1.6))+
  xlab("Predicted 3 year BCVA")+
  ylab("Observed 3 year BCVA")

Observed5 = data.frame(Observed5=testdata$BCVA_5y)
Predicted5 = data.frame(Predicted5 = predict(modelCV5, newdata = testdata))
comparison5 = as.data.frame(cbind(Observed5,Predicted5))
ggplot(aes(x=Predicted5,y=Observed5),data=comparison5)+ggtitle("Calibration")+
  geom_smooth()+
  geom_segment(aes(x=0,y=0,xend=1.6,yend=1.6))+
  xlab("Predicted 5 year BCVA")+
  ylab("Observed 5 year BCVA")



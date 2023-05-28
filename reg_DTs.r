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

#just BCVA at 3y as an example
#make the formula of selected features
form_reg <-as.formula(
  paste0('BCVA_3y ~ ',paste(colnames(traindata)[c(#selected features)],collapse = '+'))
  )

#build the model
set.seed(42)
fit_dt_reg <- rpart(
  form_reg,
  data = traindata,
  method = 'anova', #regression model
  control = rpart.control(cp = 0.005)
)
#Regression tree
printcp(fit_dt_reg)
plotcp(fit_dt_reg) #find the size of tree
fit_dt_reg_pruned <- prune(fit_dt_reg, cp=0.013)
print(fit_dt_reg_pruned)

# Use cross validation
train_control <- trainControl(method = "repeatedcv",  
                              number = 10,             
                              repeats = 3)
tune_grid = expand.grid(cp=c(0.013))
validated_tree <- train(form_reg,
                        data=traindata,                 
                        method="rpart",                    
                        trControl= train_control,          
                        tuneGrid = tune_grid,               
                        maxdepth = 5,                      
                        minbucket=5)

#the outcome of training dataset
trainpred <- predict(validated_tree, newdata = traindata)
defaultSummary(data.frame(obs = traindata$BCVA_3y, pred = trainpred))

#the outcome of testing dataset
testpred <- predict(validated_tree, newdata = testdata)
defaultSummary(data.frame(obs = testdata$BCVA_3y, pred = testpred))

#show the feature importanve
fit_dt_reg_pruned$variable.importance

#show the tree
prp(fit_dt_reg_pruned,
    type = 1,
    extra = 101,
    fallen.leaves = TRUE,
    main = 'Decision Tree')



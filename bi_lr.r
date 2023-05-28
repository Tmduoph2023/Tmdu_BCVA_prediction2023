#packages
library(MASS)
library(tidyverse)
library(corrplot)
library(caret)
library(pROC)
library(car)

#training dataset&testing dataset
traindata <- read.csv('',header=T,sep=",")
testdata <- read.csv('',header=T,sep=",")

#make the formula of selected features
form_cls <-as.formula(
  paste0('BCVA_5y_WHO ~ ',paste(colnames(traindata)[c(#selected features)],collapse = '+'))
)

#build the model
set.seed(123)
train.control <- trainControl(method= "repeatedcv", 
                            number=10,
                            repeats=3)
final_model <- train(
  form_cls,
  data = traindata,
  trControl = train.control,
  method = "glm",
  family = "binomial"
)

#the outcome of training dataset
trainpred <- predict(final_model,newdata = traindata,type = "prob")
trainroc <- roc(response = traindata$BCVA_5y_WHO, predictor = trainpred[,2])

#get the thresholds
bestp <- trainroc$thresholds[
  which.max(trainroc$sensitivities + trainroc$specificities - 1)
]
bestp

#the confusionmatrix of training dataset
trainpredlab <- as.factor(ifelse(trainpred[,2] > bestp,1,0)) 
confusionMatrix(data = trainpredlab,
                reference = traindata$BCVA_5y_WHO,
                positive = '1',
                mode = 'everything')

#the outcome of testing dataset
testpredprob <- predict(final_model, newdata = testdata, type = "prob")
testpredlab <- as.factor(ifelse(testpredprob[,2] > bestp,1,0))
confusionMatrix(data = testpredlab,
                reference = testdata$BCVA_5y_WHO,
                positive = '1',
                mode = 'everything')

#95%CI for f1_score, Sensitivity & Specificity
# create a function to compute F1-score as an example
f1 <- function(data, indices) {
  d <- data[indices,]
  testpredprob <- predict(final_model,d,type = "prob")
  predictions <- as.factor(ifelse(testpredprob[,2] > bestp,1,0))
  f1_score <- F1_Score(y_true = d$BCVA_5y_WHO, y_pred = predictions, positive = "1")  # replace F1_Score with your own F1-score function
  return(f1_score)
}
#Sensitivity <- Sensitivity(y_true = d$BCVA_5y_WHO, y_pred = predictions, positive = "1")
#Specificity <- Specificity(y_true = d$BCVA_5y_WHO, y_pred = predictions, positive = "1")

# set up
data <- testdata
# run bootstrap 
boot_result <- boot(data, f1, R = 1000)  
boot_ci <- boot.ci(boot_result, type = "bca")

#ROC & PR curve with 95%CI
target <- testdata$BCVA_5y_WHO
proba <- testpredprob[,2]
external <- as.data.frame(cbind(target, proba))
roc1 <- pROC::roc(external$target, external$proba, plot=FALSE,
                  legacy.axes=TRUE, percent=FALSE)
prcoords <- pROC::coords(roc1, "all", ret = c("threshold", "recall", "precision"), transpose = FALSE)
pr.cis <- pROC::ci.coords(roc1, prcoords$threshold, ret=c("recall", "precision"))
pr.cis <- data.frame(pr.cis[2]) 
pr.cis.df <- data.frame(x = prcoords$recall, 
                        y = prcoords$precision,
                        lower = pr.cis[, 1],
                        upper = pr.cis[, 3])
# plot precision recall coordinates along with confidence area
ggplot(prcoords, aes(recall, precision)) + 
  geom_abline(intercept = 1, slope = -1, color = "gray") +
  geom_ribbon(aes(x=pr.cis.df$x, ymin= pr.cis.df$lower, ymax= pr.cis.df$upper),alpha=0.3,fill="lightblue") +
  geom_path(aes(recall, precision), colour="black",size = 1) + # needed to connect points in order of appearance
  labs (x = "Recall 
(Sensitivity)",
        y = "Precision
(Positive predictive value)", title = "PR Curve with Confidence Intervals")+
  theme(text = element_text(size = 16))+
  theme(plot.title = element_text(size = 16, face = "bold", hjust = 0.5)) +
  theme(panel.background = element_rect(fill = "white", color = "black", size = 1),
        panel.grid.major = element_blank())+
  theme(panel.grid.minor = element_line(color = "lightgray"))+
  annotate("text", x = 0.5, y = 0.2, label = "AUC = 0.37 [0.25,0.65]", size = 6)
#plot ROC curve along with confidence area
result.boot <- boot.roc(testpredprob[,2],
                        testdata$BCVA_5y_WHO,
                        n.boot = 1000)
plot(result.boot, col = "black",  fill = "lightblue" , show.metric = 'auc') +
  geom_abline(intercept = 0, slope = 1, color = "gray") +
  labs (x = "False positive rate 
(1-Specificity)",
        y = "True positive rate
(Sensitivity)", title = "ROC Curve with Confidence Intervals")+
  theme(text = element_text(size = 12))+
  theme(plot.title = element_text(size = 16, face = "bold", hjust = 0.5)) +
  theme(panel.background = element_rect(fill = "white", color = "black", size = 1),
        panel.grid.major = element_blank())+
  theme(panel.grid.minor = element_line(color = "lightgray"))
perf(result.boot, "auc")

#plot calibration belt
Observed <- data.frame(Observed= testdata$BCVA_5y_WHO)
Observed$Observed = if_else(Observed$Observed=="1",1,0)
Predicted <- data.frame(Predicted = testpredprob[,2])
comparacao <- cbind(Observed,Predicted)
cb <- givitiCalibrationBelt(comparacao$Observed, comparacao$Predicted,devel = "external")
plot(cb, main = "Calibration Belt",
     xlab = "Predicted blindness",
     ylab = "Observed blindness"
)

#decision curve analysis
library(rmda)
dca <- function(data, outcome, predictors, xstart=0.01, xstop=0.99, xby=0.01, 
                ymin=-0.05, probability=NULL, harm=NULL,graph=TRUE, intervention=FALSE, 
                interventionper=100, smooth=FALSE,loess.span=0.10) {
  
  # LOADING REQUIRED LIBRARIES
  require(stats)
  
  # data MUST BE A DATA FRAME
  if (class(data)!="data.frame") {
    stop("Input data must be class data.frame")
  }
  
  #ONLY KEEPING COMPLETE CASES
  data=data[complete.cases(data[append(outcome,predictors)]),append(outcome,predictors)]
  
  # outcome MUST BE CODED AS 0 AND 1
  if (max(data[[outcome]])>1 | min(data[[outcome]])<0) {
    stop("outcome cannot be less than 0 or greater than 1")
  }
  # xstart IS BETWEEN 0 AND 1
  if (xstart<0 | xstart>1) {
    stop("xstart must lie between 0 and 1")
  }
  
  # xstop IS BETWEEN 0 AND 1
  if (xstop<0 | xstop>1) {
    stop("xstop must lie between 0 and 1")
  }
  
  # xby IS BETWEEN 0 AND 1
  if (xby<=0 | xby>=1) {
    stop("xby must lie between 0 and 1")
  }
  
  # xstart IS BEFORE xstop
  if (xstart>=xstop) {
    stop("xstop must be larger than xstart")
  }
  
  #STORING THE NUMBER OF PREDICTORS SPECIFIED
  pred.n=length(predictors)
  
  #IF probability SPECIFIED ENSURING THAT EACH PREDICTOR IS INDICATED AS A YES OR NO
  if (length(probability)>0 & pred.n!=length(probability)) {
    stop("Number of probabilities specified must be the same as the number of predictors being checked.")
  }
  
  #IF harm SPECIFIED ENSURING THAT EACH PREDICTOR HAS A SPECIFIED HARM
  if (length(harm)>0 & pred.n!=length(harm)) {
    stop("Number of harms specified must be the same as the number of predictors being checked.")
  }
  
  #INITIALIZING DEFAULT VALUES FOR PROBABILITES AND HARMS IF NOT SPECIFIED
  if (length(harm)==0) {
    harm=rep(0,pred.n)
  }
  if (length(probability)==0) {
    probability=rep(TRUE,pred.n)
  }
  
  
  #CHECKING THAT EACH probability ELEMENT IS EQUAL TO YES OR NO, 
  #AND CHECKING THAT PROBABILITIES ARE BETWEEN 0 and 1
  #IF NOT A PROB THEN CONVERTING WITH A LOGISTIC REGRESSION
  for(m in 1:pred.n) { 
    if (probability[m]!=TRUE & probability[m]!=FALSE) {
      stop("Each element of probability vector must be TRUE or FALSE")
    }
    if (probability[m]==TRUE & (max(data[predictors[m]])>1 | min(data[predictors[m]])<0)) {
      stop(paste(predictors[m],"must be between 0 and 1 OR sepcified as a non-probability in the probability option",sep=" "))  
    }
    if(probability[m]==FALSE) {
      model=NULL
      pred=NULL
      model=glm(data.matrix(data[outcome]) ~ data.matrix(data[predictors[m]]), family=binomial("logit"))
      pred=data.frame(model$fitted.values)
      pred=data.frame(pred)
      names(pred)=predictors[m]
      data=cbind(data[names(data)!=predictors[m]],pred)
      print(paste(predictors[m],"converted to a probability with logistic regression. Due to linearity assumption, miscalibration may occur.",sep=" "))
    }
  }
  
  # THE PREDICTOR NAMES CANNOT BE EQUAL TO all OR none.
  if (length(predictors[predictors=="all" | predictors=="none"])) {
    stop("Prediction names cannot be equal to all or none.")
  }  
  
  #########  CALCULATING NET BENEFIT   #########
  N=dim(data)[1]
  event.rate=colMeans(data[outcome])
  
  # CREATING DATAFRAME THAT IS ONE LINE PER THRESHOLD PER all AND none STRATEGY
  nb=data.frame(seq(from=xstart, to=xstop, by=xby))
  names(nb)="threshold"
  interv=nb
  
  nb["all"]=event.rate - (1-event.rate)*nb$threshold/(1-nb$threshold)
  nb["none"]=0
  
  # CYCLING THROUGH EACH PREDICTOR AND CALCULATING NET BENEFIT
  for(m in 1:pred.n){
    for(t in 1:length(nb$threshold)){
      # COUNTING TRUE POSITIVES AT EACH THRESHOLD
      tp=mean(data[data[[predictors[m]]]>=nb$threshold[t],outcome])*sum(data[[predictors[m]]]>=nb$threshold[t])
      # COUNTING FALSE POSITIVES AT EACH THRESHOLD
      fp=(1-mean(data[data[[predictors[m]]]>=nb$threshold[t],outcome]))*sum(data[[predictors[m]]]>=nb$threshold[t])
      #setting TP and FP to 0 if no observations meet threshold prob.
      if (sum(data[[predictors[m]]]>=nb$threshold[t])==0) {
        tp=0
        fp=0
      }
      
      # CALCULATING NET BENEFIT
      nb[t,predictors[m]]=tp/N - fp/N*(nb$threshold[t]/(1-nb$threshold[t])) - harm[m]
    }
    interv[predictors[m]]=(nb[predictors[m]] - nb["all"])*interventionper/(interv$threshold/(1-interv$threshold))
  }
  
  # CYCLING THROUGH EACH PREDICTOR AND SMOOTH NET BENEFIT AND INTERVENTIONS AVOIDED 
  for(m in 1:pred.n) {
    if (smooth==TRUE){
      lws=loess(data.matrix(nb[!is.na(nb[[predictors[m]]]),predictors[m]]) ~ data.matrix(nb[!is.na(nb[[predictors[m]]]),"threshold"]),span=loess.span)
      nb[!is.na(nb[[predictors[m]]]),paste(predictors[m],"_sm",sep="")]=lws$fitted
      
      lws=loess(data.matrix(interv[!is.na(nb[[predictors[m]]]),predictors[m]]) ~ data.matrix(interv[!is.na(nb[[predictors[m]]]),"threshold"]),span=loess.span)
      interv[!is.na(nb[[predictors[m]]]),paste(predictors[m],"_sm",sep="")]=lws$fitted
    }
  }
  
  # PLOTTING GRAPH IF REQUESTED
  if (graph==TRUE) {
    require(graphics)
    
    # PLOTTING INTERVENTIONS AVOIDED IF REQUESTED
    if(intervention==TRUE) {
      # initialize the legend label, color, and width using the standard specs of the none and all lines
      legendlabel <- NULL
      legendcolor <- NULL
      legendwidth <- NULL
      legendpattern <- NULL
      
      #getting maximum number of avoided interventions
      ymax=max(interv[predictors],na.rm = TRUE)
      
      #INITIALIZING EMPTY PLOT WITH LABELS
      plot(x=nb$threshold, y=nb$all,type="n",xlim=c(xstart, xstop), ylim=c(ymin, ymax), xlab="Threshold probability", ylab=paste("Net reduction in interventions per",interventionper,"patients"))
      
      #PLOTTING INTERVENTIONS AVOIDED FOR EACH PREDICTOR
      for(m in 1:pred.n) {
        if (smooth==TRUE){
          lines(interv$threshold,data.matrix(interv[paste(predictors[m],"_sm",sep="")]),col=m,lty=2)
        } else {
          lines(interv$threshold,data.matrix(interv[predictors[m]]),col=m,lty=2)
        }
        
        # adding each model to the legend
        legendlabel <- c(legendlabel, predictors[m])
        legendcolor <- c(legendcolor, m)
        legendwidth <- c(legendwidth, 1)
        legendpattern <- c(legendpattern, 2)
      }
    } else {
      # PLOTTING NET BENEFIT IF REQUESTED
      
      # initialize the legend label, color, and width using the standard specs of the none and all lines
      legendlabel <- c("None", "All")
      #legendcolor <- c(17, 8)
      legendcolor <- c('black', 'grey')
      legendwidth <- c(3, 3)
      legendpattern <- c(1, 1)
      
      #getting maximum net benefit
      ymax=max(nb[names(nb)!="threshold"],na.rm = TRUE)
      
      # inializing new benfit plot with treat all option
      plot(x=nb$threshold, y=nb$all, type="l",col="grey", lwd=1 ,xlim=c(xstart, xstop), ylim=c(ymin, ymax), xlab="Threshold probability", ylab="Net benefit")
      # adding treat none option
      lines(x=nb$threshold, y=nb$none,lwd=1,col="black")
      #PLOTTING net benefit FOR EACH PREDICTOR
      for(m in 1:pred.n) {
        if (smooth==TRUE){
          lines(nb$threshold,data.matrix(nb[paste(predictors[m],"_sm",sep="")]),col='red',lty=1,lwd = 2) 
        } else {
          lines(nb$threshold,data.matrix(nb[predictors[m]]),col='red',lty=1,lwd = 2)
        }
        # adding each model to the legend
        legendlabel <- c(legendlabel, predictors[m])
        #legendcolor <- c(legendcolor, m)
        legendcolor <- c(legendcolor, 'red')
        legendwidth <- c(3, 3)
        #legendpattern <- c(legendpattern, 2)
        legendpattern <- c(legendpattern, 1)
      }
    }
    # then add the legend
    legend("topright", legendlabel, cex=0.8, col=legendcolor, lwd=legendwidth, lty=legendpattern)
    
  }
  
  #RETURNING RESULTS
  results=list() 
  results$N=N
  results$predictors=data.frame(cbind(predictors,harm,probability))
  names(results$predictors)=c("predictor","harm.applied","probability")
  results$interventions.avoided.per=interventionper
  results$net.benefit=nb
  results$interventions.avoided=interv
  
  return(results)
  
}  
outcome <- dca(data=comparacao, outcome="Observed", predictors="Predicted", smooth="TRUE")
outcome$net.benefit

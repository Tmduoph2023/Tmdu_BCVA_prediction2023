# Tmdu_BCVA_prediction2023
Tmdu_BCVA_prediction2023 is a study to build machine learning models to predict long-term visual acuity from clinical information in highly myopic eyes. 
The training platforms were the NVIDIA GeForce GTX 1080 Graphics Card (NVIDIA Corporation, CA, US) and the Ubuntu 18.04 LTS (Canonical, London, UK). All programs were performed by R version 4.1.2. The used packaged were listed in each model files.



Regression models: 

We included the following models to predict BCVA at 3 and 5 years: Linear Regression, Support Vector Machines (SVM), Decision Trees (DTs), Random Forest (RF), and eXtreme Gradient Boosting (XGboost). 

Please see files reg_lm.r, reg_svm.r, reg_DTs.r, reg_RF.r and reg_xgboost.r.



Binary classification model:

When predicting the risk of VI at 5 years, five models were included: Logistic Regression, SVM, DTs, RF, and XGboost.

Please see files bi_lm.r, bi_svm.r, bi_DTs.r, bi_RF.r and bi_xgboost.r.

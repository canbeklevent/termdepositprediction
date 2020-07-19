# free memory
rm(list = ls())
gc()

getwd()
setwd("C:/Users/Simba/Desktop/")

#importing dataset
df <- read.csv("term-deposit-marketing-2020.csv",header = TRUE)

dim(df) #40000 rows, 14 columns (attributes)

#checking if there is any column with null value
colSums(is.na(df)) #there is no null column

#checking the structure to see column types and values
str(df)

install.packages("gmodels")
library(gmodels)
CrossTable(df$y) #dependent variable has 37104 "no" and 2896 "yes" values, unbalanced binary data

df$y = as.factor(ifelse(df$y =="yes",1,0))

#grouping ages
df$age =if_else(df$age > 60, "high", if_else(df$age > 30, "mid", "low"))


#converting chr typed columns into factors
library(dplyr)
df <- df %>% mutate_if(is.character,as.factor)


#structure of resulted df to see types and values
str(df)
#summary of attributes with different value distributions
summary(df)

#################################### EDA ######################################

library(funModeling)

#cross plots between some independent attributes and dependent variable y which is deposit decision
cross_plot(data=df, input="job", target="y") #15.6% of students preferred term deposit with highest ratio

cross_plot(data=df, input="age", target="y") #39% of old people preferred term deposit with highest ratio

cross_plot(data=df, input="education", target="y") #9% of people with highest education level preferred term deposit with highest ratio but the ratios are closer to each other

cross_plot(data=df, input="month", target="y") #surprisingly, in october there were 80 offers which are resulted with 49 "yes" decisions which is the highest yes ratio between months

cross_plot(data=df, input="marital", target="y") #ratios are similiar with highest ratio of 9.4% for single people

cross_plot(data=df, input="default", target="y") #most of the people does not have any deposit in default and this attribute does not clearly affect the term deposit decision

cross_plot(data=df, input="contact", target="y")

#some chi-square tests to test association btw binary independent variables and y

chisq.test(df$default, df$y) #p value is higher than 0.05 (0.21) for confidence level of 95%, it means default attribute does not have clear association with y

chisq.test(df$housing, df$y) #a significant result with p value very lower than 0.05. for confidence level of 95% there is a clear association btw housing and y

chisq.test(df$loan, df$y) #a significant result with p value very lower than 0.05. for confidence level of 95% there is a clear association btw loan and y

#problem about "duration" atribute: it highly affects the result of y: 0 for lower values of duration

cross_plot(data=df, input="duration", target="y") #after duration > 364 s, the graph shows a sharp change for "yes" value of y

#Pearson correlation test results also show that there is a correlation btw duration and y with p-value very lower than 0.05 and cor coef 0.46
cor(as.numeric(df$y), df$duration, method = c("pearson"))
cor.test(as.numeric(df$y), df$duration, method=c("pearson"))

#Also, duration is an attribute that can not be known before making a call. So to create more realistic models, i will exclude this attribute from ML models for prediction

#using var_rank_info, we can rank the attributes importance for dependent variable y based on information gain
var_imp<-var_rank_info(df, "y")

# Plotting 
ggplot(var_imp, 
       aes(x = reorder(var, gr), 
           y = gr, fill = var)
) + 
  geom_bar(stat = "identity") + 
  coord_flip() + 
  theme_bw() + 
  xlab("") + 
  ylab("Importance of Attributes based on Term Deposit decisions)"
  ) + 
  guides(fill = FALSE)
#Plot also shows us that "duration" is the most important attribute w.r.t "y" value based on information theory



###################################### PREDICTION MODELS ################################


#setting the seed number
set.seed(123)

#getting a sample of data rows, %80 for training dataset, %20 for test dataset
library(caret)
t_rows = createDataPartition(df$y,
                          times = 1,
                          p = 0.8,
                          list = F)
train_df = df[t_rows,]
test_df = df[-t_rows,]

#################################### LOGISTIC REGRESSION ################################

#duration is excluded from the logit model for the reasons stated above
logit_model <-glm(y~.-duration,data = train_df,family = "binomial")
summary(logit_model)

#According to the results, some attributes are insignificant such as default and job groups excluding being a housemaid
#"low" and "mid" levels of ages show negative relation with term deposit preference "y" with p-values very lower than 0.001. 
#In job types, only related one is being a housemaid. The people which are housemaids show negative relation with y with p-value close to 0.02.
#Married people show negative relation with dependent value "y" based on p-value very lower than 0.001.
#Highest education level (tertiary) shows a sginification positive relation with p-value close to 0.001.
#As expected, balance shows a significantly positive affect on variable "y" with p-value close to 0.005.
#The people who own house loan do not prefer term deposits with a very lower p-value.It is similar for people who own personal loan.
#Telephone as contact type shows a negative affect on variable "y" with a very lower p-value.
#Last days of months are more preferred for term deposit approvals.
#Calling the client at month october positively affect the deposit choice while calling at months feb,jan,jul,mayinov negatively affects.
#Number of contacts (campaign) is a significant parameter with negative affect on term deposit choice with p-value very lower than 0.001.


#Prediction and validating the model on testing dataset

#measure of validation model goodness
test_prediction <- predict.glm(logit_model, newdata=test_df, type= "response")

#using threshold 0.5 for "yes" decision
test_prediction[test_prediction > 0.5] <- 1
test_prediction[test_prediction <= 0.5] <- 0

cm_pred <- table(test_prediction, test_df$y)
accuracy = (cm_pred[1,1] + cm_pred[2,2]) / (cm_pred[1,1] + cm_pred[2,2] + cm_pred[1,2] + cm_pred[2,1])

print(accuracy)
print(cm_pred) 

#Result: very good accuracy rate: 0.927 with very lower true positive prediction: 18/ 579

#Deciding on a good threshold here is important because we have a very unbalanced distributing of "yes" and "no" values of y 
#which makes reaching a good accuracy scores on models easier.
#Because in dataset, there are very little numbers of people who says "yes" to term deposit
#it is important to correctly predict customers who are actually willing to take deposit which are true positive group.
#So my focus here is to get a good rate of true positives while also ensuring low false negative rates.

#To do that, i will use F score because it provides a score that balances recall and precision scores 
#which are important metrics on unbalanced classification as i have.
#precision score indicates the number of positive class predictions that are actually in positive class
#recall score indicates the number of positive class predictions resulted from all positives in dataset

precision_score <- cm_pred[2,2]/(sum(cm_pred[2,]))
recall_score <- cm_pred[2,2]/(sum(cm_pred[,2]))
print(precision_score) #0.48
print(recall_score) #0.03
F1<- 2*precision_score*recall_score/(precision_score+recall_score)
print(F1) #0.058
#When "yes" decision threshold is 0.5, we get very low F score which is an indicator of model goodness based on predicting true positives

#Now, we will calculate F score for threshold 0.2:

#measure of validation model goodness
test_prediction_2 <- predict.glm(logit_model, newdata=test_df, type= "response")

#using threshold 0.2 for "yes" decision
test_prediction_2[test_prediction_2 > 0.2] <- 1
test_prediction_2[test_prediction_2 <= 0.2] <- 0

cm_pred_2 <- table(test_prediction_2, test_df$y)
accuracy_2 = (cm_pred_2[1,1] + cm_pred_2[2,2]) / (cm_pred_2[1,1] + cm_pred_2[2,2] + cm_pred_2[1,2] + cm_pred_2[2,1])

print(accuracy_2)
print(cm_pred_2) 

precision_score_2 <- cm_pred_2[2,2]/(sum(cm_pred_2[2,]))
recall_score_2 <- cm_pred_2[2,2]/(sum(cm_pred_2[,2]))
print(precision_score_2) #0.38
print(recall_score_2) #0.16
F1<- 2*precision_score_2*recall_score_2/(precision_score_2+recall_score_2)
print(F1) #0.23
#When "yes" decision threshold is 0.2, F1 is score is now higher but it is still not much good 
#But as we continue to decrease the threshold, recall_Score may get higher but it will decrease precision rate
#and will result in lower values of F.


#5-fold cross validation for logit model

#implementing a vector to store all 5 accuracy score
acc <- rep(0,5) 

#for loop to construct the tree model 5 times 
for (i in 1:5) {
  #take 5 different sample from data
  t_rows = createDataPartition(df$y,
                               times = 1,
                               p = 0.8,
                               list = F)
  
  #divide the data 5 times to training and test datasets
  train_df = df[t_rows,]
  test_df = df[-t_rows,]
  
  #duration is excluded from the logit model for the reasons stated above
  #construct the model 5 times with different training datasets
  logit_model <-glm(y~.-duration,data = train_df,family = "binomial")
  summary(logit_model)
  
  #measure of validation model goodness
  test_prediction <- predict.glm(logit_model, newdata=test_df, type= "response")
  
  #using threshold 0.2 for "yes" decision
  test_prediction[test_prediction > 0.2] <- 1
  test_prediction[test_prediction <= 0.2] <- 0
  
  #construct the confusion matrix
  cm_pred <- table(test_prediction, test_df$y)
  acc[i] = (cm_pred[1,1] + cm_pred[2,2]) / (cm_pred[1,1] + cm_pred[2,2] + cm_pred[1,2] + cm_pred[2,1])
  
  precision_score[i] <- cm_pred[2,2]/(sum(cm_pred[2,]))
  recall_score[i] <- cm_pred[2,2]/(sum(cm_pred[,2]))
  F1[i]<- 2*precision_score[i]*recall_score[i]/(precision_score[i]+recall_score[i])
  
  
}

#print the vector indices which are the different accuracy scores we calculated from cross validation which are ranged between 0.918 and 0.924
print(acc)

#calculated the mean of these accuracy scores as 0.92.
mean(acc)

#print mean of F scores
print(mean(F1)) #0.265

#Result: although getting higher values of accuracy scores, because of having an unbalanced distribution of y values
#i ended up with lower values of F scores. When i changed "yes" decision threshold to 0.2, resulted F score became 0.23 with accuracy score 0.92.

##################################### DECISION TREE PREDICTION ###############################

#constructing the model using training dataset
install.packages("party")
install.packages("partykit")
library (ctree)

formula <- y~.-duration
tree_model = ctree(formula,data = train_df)

print(tree_model)
plot(tree_model)

#using the classification model to predict on testing data
testpredictions = predict(tree_model,test_df,method = "class")

#7982 "No" and 17 "Yes" prediction from 7999 rows
summary(testpredictions)

#constructing the confusion matrix and calculating the accuracy score of our model for testing data.
cm <- table(testpredictions,test_df$y)
accuracy = (cm[1,1] + cm[2,2]) / (cm[1,1] + cm[2,2] + cm[1,2] + cm[2,1])

#Accuracy score calculated as 0.928 with 7414 true "No" prediction and 11 true "Yes" Prediction from 8000 rows.
print(cm)
print(accuracy)

#5-fold cross validation for decision tree:
#implementing a vector to store all 5 accuracy score
acc <- rep(0,5) 

#for loop to construct the tree model 5 times 
for (i in 1:5) {
  #take 5 different sample from data
  t_rows = createDataPartition(df$y,
                               times = 1,
                               p = 0.8,
                               list = F)
  
  #divide the data 5 times to training and test datasets
  train_df = df[t_rows,]
  test_df = df[-t_rows,]
  
  #construct the model 5 times with different training datasets
  formula <- y~.-duration
  tree_model = ctree(formula,data = train_df)
  
  #using the classification model to predict on testing data
  testpredictions = predict(tree_model,test_df,method = "class")

  
  #construct the confusion matrix
  cm_pred <- table(testpredictions, test_df$y)
  acc[i] = (cm_pred[1,1] + cm_pred[2,2]) / (cm_pred[1,1] + cm_pred[2,2] + cm_pred[1,2] + cm_pred[2,1])
  
  
}
#print the vector indices which are the different accuracy scores we calculated from cross validation which are ranged between 0.926 and 0.928
print(acc)

#calculated the mean of these accuracy scores as 0.927.
mean(acc)


#Again not surprisingly, true positive prediction rate is very low with a very good accuracy rate because of high true negative rates.

#Conclusion from the prediction models with Logistic Reg and Decision Tree:
#Models tend to get biased toward the majority class which is "no" decision for term deposit

#Because of that, now i will try to overcome the problem of imbalanced classification data 
#using sampling methods to get a more balanced data with CART and 5-fold cv

library(caret)

control <- trainControl(method = "repeatedcv", repeats = 5,
                     classProbs = TRUE,
                     summaryFunction = twoClassSummary)

#undersampling
down_training <- downSample(x = train_df[, -ncol(train_df)],
                            y = train_df$y)

#checking the proportion the target class after down sampling
prop.table(table(down_training$Class)) #yes 0.5, no 0.5 in 4634 observations

#oversampling
up_training <- upSample(x = train_df[, -ncol(train_df)],
                            y = train_df$y)

#checking the proportion the target class after up sampling
prop.table(table(up_training$Class)) #yes 0.5, no 0.5 in 59368 observations


####Model development with under sampled data
metric <- "auc"
set.seed(7)
mtry <- sqrt(ncol(down_training))
tunegrid <- expand.grid(.mtry=mtry)
rf_default <- train(Class~., data=down_training, method="rf", metric=metric, tuneGrid=tunegrid, trControl=control)
print(rf_default) #ROC 0.936

##prediction on unseen data using the trained rf_default model
rf_response <- predict(rf_default, test_df, type = "prob")
rf_response <- rf_response[,2]

#using AUC-ROC score to determine model fit
pred_rf <- prediction(rf_response, test_df$y)
perf_rf <- performance(pred_rf, measure = "tpr", x.measure = "fpr")
plot(perf_rf)
auc(test_df$y, rf_response)#0.934

####Model development with over sampled data
metric <- "auc"
set.seed(7)
mtry <- sqrt(ncol(up_training))
tunegrid <- expand.grid(.mtry=mtry)
rf_default <- train(Class~., data=up_training, method="rf", metric=metric, tuneGrid=tunegrid, trControl=control)
print(rf_default) 

##prediction on unseen data using the trained rf_default model
rf_response <- predict(rf_default, test_df, type = "prob")
rf_response <- rf_response[,2]

#using AUC-ROC score to determine model fit
pred_rf <- prediction(rf_response, test_df$y)
perf_rf <- performance(pred_rf, measure = "tpr", x.measure = "fpr")
plot(perf_rf)
auc(test_df$y, rf_response)







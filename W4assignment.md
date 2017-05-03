---
title: "Practical Machine Learning Project : Prediction Assignment"
author: "madhav"
date: "May 1, 2017"
output: html_document
---


##Background of data

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement - a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: http://groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset).

##sources of Data

The training data for this project are available here:

https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv

The test data are available here:

https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv

The data for this project come from this source: http://groupware.les.inf.puc-rio.br/har. If you use the document you create for this class for any purpose please cite them as they have been very generous in allowing their data to be used for this kind of assignment.

##What you should submit

The goal of your project is to predict the manner in which they did the exercise. This is the "classe" variable in the training set. You may use any of the other variables to predict with. You should create a report describing **how you built your model, how you used cross validation, what you think the expected out of sample error is**, and why you made the choices you did. You will also use your prediction model to predict 20 different test cases.


###loading data:
we can donload data using the chunk below
```{r}
#training data
download.file(url = "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv", destfile = "./pml-training.csv", method = "curl")
#testing data
download.file(url = "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv", destfile = "./pml-testing.csv", method = "curl")

```

as i have already downloaded the data,i will now load them.use "na.strings" to ensure that miscellaneous NA, #DIV/0! and empty fields are represented as NA.
```{r}
setwd("E:\\course era\\practical Ml\\w4")#setting directory
train<-read.csv("pml-training.csv",na.strings=c("NA","#DIV/0!",""))
test<-read.csv("pml-testing.csv",na.strings=c("NA","#DIV/0!",""))
dim(train)
dim(test)#can check the number of features given
str(train, list.len=20)
table(is.na(train))#gives number of NA in complete data.
table(is.na(train$classe))
```
from the above we are clear that there are many NA in the data, and even first six of the features/independent variables given are not that useful in model building so we can discard them.

```{r}
train <- train[, 7:160]#removing unwanted coloumns.
test  <- test[, 7:160]
```
##we are here removing the coloumns that have more number of NA(based on observations in "classe")
```{r}
out_data  <- apply(!is.na(train), 2, sum) > 19621  # which is the number of observations
train <- train[, out_data]
test  <- test[, out_data]
dim(train)
dim(test)#can check the number of features given

```
##Partitioning the Dataset
here we split the train in the ratio 60:40,so that we can perform training and cross validation,evaluation respectively.we can estimate the sample error by doing so.

```{r}
set.seed(3141592)
library(caret)
library(ggplot2)
inTrain <- createDataPartition(y=train$classe, p=0.60, list=FALSE)
train1  <- train[inTrain,]
train2  <- train[-inTrain,]
dim(train1)
dim(train2)
```
 we can see that train1:-11776x54,train2:-7846x54.we check here for near zero covariates and discard them.
```{r}
nzv_cols <- nearZeroVar(train1)
if(length(nzv_cols) > 0) {
  train1 <- train1[, -nzv_cols]
  train2 <- train2[, -nzv_cols]
}
dim(train1)
dim(train2)
```
as observed no near zero covariates were seen maybe due to our data cleaning done on features with NA'S.we are now ready to build a model.before that we will be checking the most important  of all features.


```{r}
library(randomForest)
## randomForest 4.6-10
## Type rfNews() to see new features/changes/bug fixes.
set.seed(3141592)
fitModel <- randomForest(classe~., data=train1, importance=TRUE, ntree=100)
varImpPlot(fitModel,main = "varImpPlot")
imp<-importance(fitModel)
imp_f<-imp[,7]>200#here i have selected top features using gini index.here 200 is                   #being picked up randomly
imp_f_names<-imp_f[imp_f=="TRUE"]
imp_f_names_names<-names(imp_f_names)
```
now we will check for corelations between these feature that have gini value>200.
```{r}
corrmatrix = cor(train1[,imp_f_names_names])
library(arm)
corrplot(corrmatrix)#we can see the corelated features in the plot.
diag(corrmatrix) <- 0
which(abs(corrmatrix)>0.75, arr.ind=TRUE)#co related features with value more than 0.75.
which(abs(corrmatrix)>0.90, arr.ind=TRUE)#co related features with value more than 0.90.
```
by observing above result it is clear that 
yaw_belt&roll_belt,
accel_belt_z&roll_belt,
accel_belt_z&yaw_belt,
magnet_belt_z&magnet_belt_y,
magnet_dumbbell_y&magnet_dumbbell_x are corelated.

if we observe the varImpPlot we can clearly see that  yaw_belt&roll_belt show high priority,here should apply pca analysis for the corelated feature but iam going on without applying pca.so we can discard "accel_belt_z" as it has corelation with both,magnet_belt_z,magnet_dumbbell_x are to be discarded.for ease yaw_belt has been discarded.


```{r}
train1<-train1[,-which(names(train1) %in% c("accel_belt_z","magnet_belt_z","magnet_dumbbell_x","yaw_belt"))]

```

##Building the Decision Tree Model
we will first build decision tree model 
```{r}
library(rpart)
library(rpart.plot)
dtModel <- rpart(classe~., data=train1, method="class")
prp(dtModel, box.palette="auto")#plot of decision tree,we can see that the first feature selected is rollbelt, this reiterates its importance in the data.

```

we are now evaluating the decision tree model using confusion matrix.
here we can see that the accuracy is around 75% only."Kappa : 0.6815" suggest that there is a moderate level of agreement between model and train2.


```{r}
set.seed(12345)
prediction <- predict(dtModel, newdata = train2, type = "class")
caret::confusionMatrix(prediction, train2$classe)
```
##out of sample error estimation
this gives the out of sample error as 25.03%.
```{r}
sum(prediction!= train2$classe) / length(train2$classe)
```
##Building the Random forest model
we will now build the random forest model on the same data on which we built decision tree.
```{r}
set.seed(12345)
modFitRF <- randomForest(classe ~ ., data = train1, ntree = 1000)

```

we are now evaluating the random forest model using confusion matrix.
here we can see that the accuracy is around 99.59% only."Kappa : 0.9948" suggest that there is a very good level of agreement between model and train2.we compare the results from the decision tree and random forest.
```{r}
prediction <- predict(modFitRF, newdata = train2, type = "class")
confusionMatrix(prediction, train2$classe)
```
##out of sample error estimation
number of missclasifications dicvided by total predictions gives out sample error.here it is only " 0.40785".The out-of-sample error rate has fallen to 0.40785 from that of decision tree's 25%.
```{r}
sum(prediction!= train2$classe) / length(train2$classe)
```

##Predicting on the Testing Data (pml-testing.csv)
now lets apply our models on the testing data given and which is stored in train(till now we did not used).
###Prediction on Decision Tree
```{r}
DTprediction <- predict(dtModel,test , type = "class")
DTprediction

```

###Prediction on random forest

```{r}
rfprediction <- predict(modFitRF,test , type = "class")
rfprediction

```

#Submission file
By comparing we found  Random Forest model is very accurate, about 99%. Because of that we could expect nearly all of the submitted test cases to be correct.

Prepare the submission file.
```{r}
pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}

pml_write_files(rfprediction)

```
##conclusion
here we have built decision tree model and random forest model on the given data, and have got about 99% accuracy in random forest model.while refering this donot over look the assumptions made.


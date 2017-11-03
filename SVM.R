setwd("C:/Users/lingq/Google Drive/4335 Machine Learning/midterm project")
setwd("C:/Users/Administrator/Google Drive/4335 Machine Learning/midterm project")
#Load data
source("load_mnist.R")
Train = data.frame(y=as.factor(train$y), train$x/255)
Test = data.frame(y=as.factor(test$y), test$x/255)


train <- read.csv("mnist_train.csv",head=F)
test <- read.csv("mnist_test.csv",head=F)

Train = data.frame(y=as.factor(train[,1]), train[,-1]/255)
Test = data.frame(y=as.factor(test[,1]), test[,-1]/255)
#simple SVM
require(e1071)
trainNum=c(10000,20000,30000,40000)
for(i in 1:length(trainNum)){
  num=trainNum[i]
  set.seed(1)
  trainsub=Train[sample(1:nrow(Train),num),]
svm=svm(y~.,data=trainsub,kernel='radial')
svm.table=table(Test$y,predict(svm,newdata=Test))
print(svm.table)
accuracy=sum(diag(svm.table))/nrow(Test)
print(accuracy)
}



set.seed(1)
trainIndex=sample(1:nrow(inTrain),floor(0.7*nrow(inTrain)))
training = inTrain[trainIndex,]
cv = inTrain[-trainIndex,]
trainNum =c(1000,2000,3000)
# trainNum = c(10000, 12500, 15000, 17500, 20000, 25000, 30000, 42000)
len <- length(trainNum)
accuracy <- data.frame(trainNum=trainNum,radial=rep(NA,len),linear=rep(NA,len),sigmoid=rep(NA,len),polynomial=rep(NA,len))
time=data.frame(trainNum=trainNum,time=rep(NA,len))

for(i in length(trainNum)){
n.training=trainNum[i]
set.seed(123)
trainingsub=training[sample(1:nrow(training),n.training),]
time0 <- Sys.time()
#SVM
require(e1071)
#radial
radial.tune<-tune.svm(y~.,data=trainingsub,kernel='radial')
radial.table <- table(cv$y,predict(radial.tune$best.model,newdata=cv))
radial.table
accuracy$radial[i] <- sum(diag(radial.table))/nrow(cv)
write.table(radial.table,paste0("SVM_radial_",n.training,".csv"),sep = ",")

#linear
linear.tune<-tune.svm(y~.,data=trainingsub,kernel='linear')
linear.table <- table(cv$y,predict(linear.tune$best.model,newdata=cv))
linear.table
accuracy$linear[i] <- sum(diag(linear.table))/nrow(cv)
write.table(linear.table,paste0("SVM_linear_",n.training,".csv"),sep = ",")

#sigmoid
sigmoid.tune<-tune.svm(y~.,data=trainingsub,kernel='sigmoid')
sigmoid.table <- table(cv$y,predict(sigmoid.tune$best.model,newdata=cv))
sigmoid.table
accuracy$sigmoid[i] <- sum(diag(sigmoid.table))/nrow(cv)
write.table(sigmoid.table,paste0("SVM_sigmoid_",n.training,".csv"),sep = ",")

#polynomial
trainingsub[,-1] <- trainingsub[-1]*255
cv[,-1] <- cv[,-1]*255
polynomial.tune<-tune.svm(y~.,data=trainingsub,kernel='polynomial')
polynomial.table <- table(cv$y,predict(polynomial.tune$best.model,newdata=cv))
polynomial.table
accuracy$polynomial[i] <- sum(diag(polynomial.table))/nrow(cv)
write.table(polynomial.table,paste0("SVM_polynomial_",n.training,".csv"),sep = ",")

time$time <- Sys.time()-time0
}


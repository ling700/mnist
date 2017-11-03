setwd("C:/Users/lingq/Google Drive/4335 Machine Learning/midterm project")
source("load_mnist.R")
Train = data.frame(y=as.factor(train$y), train$x)
Test= data.frame(y=as.factor(test$y),test$x)
train.attr <- Train[,-1]
train.label <- Train[,1]

#eigenvalue 
V <- cov(train.attr/255)#calculate the covirance matrix
eV <- eigen(V)#do eigen decomposion with covirance matrix
eigenvector <- eV$vectors
eigenvalue <- eV$values
opar=par(no.readonly = TRUE)
par(mfrow=c(2,4),mar=c(2,2,1,1))
for(i in 1:20){
  show_digit(eigenvector[,i])
}
par(opar)

eigenvalue[eigenvalue>0.8]
cumsum <- cumsum(eigenvalue/sum(eigenvalue))#c
cumsum[cumsum<0.95]
plot(cumsum[cumsum<0.95],xlab="number of components",ylab="cumulative contribution rate")
abline(h=0.8,col="red")
abline(v=length(cumsum[cumsum<0.8]),col="red")
text(43,0.1,labels="43")

n.prin=length(cumsum[cumsum<0.8])+1
trans <- eigenvector[,1:n.prin]
train.princomp<-data.matrix(train.attr)%*%trans
test.princomp<-data.matrix(test[,-1])%*%trans


#kmpp
kmpp.prin.wss <- (nrow(train.princomp)-1)*sum(apply(train.princomp,2,var))
kmpp.prin.cluster.table=list()
kmpp.prin.accuracy=rep(NA,30)
for (k in 2:30){
  kmpp.prin.cluster <- kmeanspp(train.princomp,k)
  kmpp.prin.wss[k] <- sum(kmpp.prin.cluster$withinss)
  kmpp.prin.cluster.table[[k]] <- table(train.label,kmpp.prin.cluster$cluster)
  kmpp.prin.accuracy[k] <- cluster_accuracy(train.label,kmpp.prin.cluster$cluster)
  
}

plot(1:30, kmpp.prin.wss, type="b", xlab="Number of Clusters",
     ylab="Within groups sum of squares",
     main="Assessing the Optimal Number of Clusters with the Elbow Method")

plot(1:30, kmpp.prin.accuracy, type="b", xlab="Number of Clusters",
     ylab="Accuracy",
     main="Accuracy of Kmeans++ based oh PCA")

setwd("C:/Users/lingq/Google Drive/4335 Machine Learning/midterm project")
source("load_mnist.R")

Train = data.frame(y=as.factor(train$y), train$x)
Test= data.frame(y=as.factor(test$y),test$x)
train.attr <- Train[,-1]
train.label <- Train[,1]
kmeans.cluster <- kmeans(train.attr, 16)
kmeans.cluster$size
table(train.label,kmeans.cluster$cluster)

wss <- (nrow(train.attr)-1)*sum(apply(train.attr,2,var))
kmeans.cluster.table=list()
accuracy=rep(NA,30)
for (k in 1:30) {
  kmeans.cluster <- kmeans(train.attr,centers=k)
  wss[k] <- sum(kmeans.cluster$withinss)
  kmeans.table <- as.matrix(table(train.label,kmeans.cluster$cluster))
  inferred_label=rep(NA,k)
  for(i in 1:k){
    inferred_label[i]=names(which.max(kmeans.table[,i]))
  }
  colnames(kmeans.table) <- inferred_label
  
  for(i in 1:nrow(Train)){
    cluster=kmeans.cluster$cluster[i]
    Train$infer[i]=inferred_label[cluster]
  }
  kmeans.cluster.table[[k]] <- table(Train$infer,Train$y)
  accuracy[k]=sum(Train$infer==Train$y)/nrow(Train)
}

plot(1:30, wss, type="b", xlab="Number of Clusters",
     ylab="Within groups sum of squares",
     main="Assessing the Optimal Number of Clusters with the Elbow Method")

plot(1:30, accuracy, type="b", xlab="Number of Clusters",
     ylab="Accuracy",
     main="Assessing the Optimal Number of Clusters with the Elbow Method")
kmeanspp <- function(data, k) { 
  n <- nrow(data) 
  C <- numeric(k) 
  C[1] <- sample(1:n, 1) 
  
  for (i in 2:k) { 
    dm <- distmat(data, data[C, ]) 
    pr <- apply(dm, 1, min)
    pr[C] <- 0 
    C[i] <- sample(1:n, 1, prob = pr) 
  } 
  
  kmeans(data, data[C, ]) 
} 



clusters <- hclust(dist(Train), method = 'average')
plot(clusters)

require(dbscan)
Train.mat <- as.matrix(Train[1:1000,-1])
kNNdistplot(Train.mat,k=10)
res <- dbscan(Train.mat, eps = 2000, minPts = 10)
res


data(iris)
iris <- as.matrix(iris[,1:4])
kNNdistplot(iris, k = 5)
abline(h=.4, col = "red", lty=2)
res <- dbscan(iris, eps = .4, minPts = 5)
res
pairs(iris, col = res$cluster + 1L)
## use precomputed frNN
fr <- frNN(iris, eps = .4)
dbscan(fr, minPts = 5)
## example data from fpc
set.seed(665544)
n <- 100
x <- cbind(
  x = runif(10, 0, 10) + rnorm(n, sd = 0.2),
  y = runif(10, 0, 10) + rnorm(n, sd = 0.2)
)
res <- dbscan(x, eps = .3, minPts = 3)
res
## plot clusters and add noise (cluster 0) as crosses.
plot(x, col=res$cluster)
points(x[res$cluster==0,], pch = 3, col = "grey")
hullplot(x, res)

# En este código hago las gráficas base correspondientes a nuestras features. 

train <- read.csv("Data/train.csv");
attach(train);

my_features <-  names(train);

summary(train)

listOfColors<-rainbow(30)

n<-dim(train)[1]

K <- dim(train)[2]

for(k in 3:K){
  if(is.factor(train[,k])){
    frecs <- table(train[,k]);
    proportions <- frecs/n;
    imagename = paste("Data/Pie" , names(train)[k],sep = "_")
    imagename = paste(imagename, "png",sep = ".")
    png(imagename, width = 800, height = 600)
    pie(frecs, cex=0.6, main=paste("Pie of", names(train)[k]))
    dev.off()
    imagename = paste("Data/Barplot" , names(train)[k],sep = "_")
    imagename = paste(imagename, "png",sep = ".")
    png(imagename, width = 800, height = 600)
    barplot(frecs, las=3, cex.names=0.7, main=paste("Barplot of", names(train)[k]), col=listOfColors)
    dev.off()
    print(frecs)
    print(proportions)
  }else{
    hist(train[,k], main=paste("Histogram of", names(train)[k]))
    boxplot(train[,k], horizontal=TRUE, main=paste("Boxplot of", names(train)[k]))
    print(summary(train[,k]))
    print(paste("sd: ", sd(train[,k])))
    print(paste("vc: ", sd(train[,k])/mean(train[,k])))
  }
}

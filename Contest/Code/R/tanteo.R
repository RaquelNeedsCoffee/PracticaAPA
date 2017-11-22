# El objetivo de este fichero es tantear si el problema es viable para el trabajo
# Nuestro problema consiste en, dados datos sobre los usuarios, decidir si después de escuchar
# una canción la volverá a escuchar repetidamente. 


# Leo los datos

# Metadatos sobre las canciones y usuarios

# read.csv("Data/members.csv");
# read.csv("Data/sample_submission.csv");
# read.csv("Data/song_extra_info.csv");
# read.csv("Data/songs.csv");

# Train y test
<<<<<<< master
#train <- read.csv("Data/test.csv");
train <- read.csv("Data/train.csv");

# Le quito valores para que no pete. 

## 50% of the sample size
smp_size <- floor(0.001 * nrow(train))

## set the seed to make your partition reproductible
set.seed(123)
train_ind <- sample(seq_len(nrow(train)), size = smp_size)

train <- train[train_ind, ]

# Lo que queda lo particiono en test y train

## 75% of the sample size
smp_size <- floor(0.75 * nrow(train))

## set the seed to make your partition reproductible
set.seed(123)
train_ind <- sample(seq_len(nrow(train)), size = smp_size)

train <- train[train_ind, ]
test <- train[-train_ind, ]

write.csv(train, file = "Data/ReducedTrain.csv")
write.csv(test, file = "Data/ReducedTest.csv")

train = read.csv("Data/ReducedTrain.csv")
test = read.csv("Data/ReducedTest.csv")
=======
#read.csv("Data/test.csv");
train <- read.csv("Data/train.csv");

attach(train);
# Hace falta preprocessing?

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
library(Amelia)
missmap(train, main = "Missing values vs observed")


>>>>>>> Estudio inicial de los datos del concurso
# Ahora hay que elegir entre 3 de las siguientes opciones: 
#  logistic regression, multinomial regression
#(single-layer MLP), LDA, QDA, RDA, Naive Bayes, nearest-neighbours, linear
#SVM, quadratic SVM
<<<<<<< master

# Pruebo un lm 
model <- glm(target ~.,family=binomial(link='logit'),data=train)
gc()
summary(model)
anova( model, test="Chisq")
=======
>>>>>>> Estudio inicial de los datos del concurso

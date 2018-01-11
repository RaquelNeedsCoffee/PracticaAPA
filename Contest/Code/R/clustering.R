# We load the cleaned data and attach it: 
dd <- read.csv("./Datasets/pokemonProcessed.csv", header=T, sep = ",");
attach(dd)

#We select the numerical variables and save it in numerical.variables

names(dd)
numerical.variables <- data.frame (Total,Sp_Atk,Sp_Def,Speed,Attack,Defense,HP, Catch_Rate,Height_m,Weight_kg)
dim(numerical.variables)
summary(numerical.variables)
# HIERARCHICAL CLUSTERING

d  <- dist(numerical.variables)
h1 <- hclust(d,method="ward.D")
plot(h1)

# We take 4 clases

nc = 4
c1 <- cutree(h1,nc)
table(c1)

# To interpret the resoults we compare the means of the four clases for every variable

cdg <- aggregate(as.data.frame(numerical.variables),list(c1),mean)
cdg

pairs(cdg[,2:11], col=cdg[,1])

plot(cdg[,'Total'], cdg[,'Catch_Rate'],col=cdg[,1],main="Clustering of pokemon data in 4 classes")
legend("topright",c("class1","class2","class3", "class4"),pch=1,col=c(1:4))



# We compare Catch_Rate with Total

plot(Total, Catch_Rate,col=c1,main="Clustering of pokemon data in 4 classes")
legend("topright",c("class1","class2","class3", "class4"),pch=1,col=c(1:4))

names <- c("isLegendary","Total", "Catch_Rate")
pairs(numerical.variables[,names], col=c1)

# Now we test the quality of our partition

Bss <- sum(rowSums(cdg^2)*as.numeric(table(c1)))

Ib4 <- 100*Bss/Tss
Ib4


# Now we use Gower mixed distance to deal 
# simoultaneously with numerical and qualitative data

library(cluster)

#dissimilarity matrix

dissimMatrix <- daisy(dd, metric = "gower", stand=TRUE)

distMatrix<-dissimMatrix^2

h2 <- hclust(distMatrix,method="ward.D")  

plot(h2)

# Now is better to take 3 classes 
c2 <- cutree(h2,3)

#class sizes 
table(c2)

#comparing with other partitions
table(c1,c2)


# We do some boxplots to study the clusters selected

boxplot(Total~c2, horizontal=TRUE)


boxplot(Catch_Rate~c2, horizontal=TRUE)



names <- c("isLegendary","Total", "Catch_Rate")
pairs(dd[,names], col=c2)

cdg <- aggregate(as.data.frame(numerical.variables),list(c2),mean)
cdg



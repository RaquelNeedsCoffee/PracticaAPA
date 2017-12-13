library(data.table) # 
library(FactoMineR)
library(factoextra)

dt <- fread("Contest/Data/samples/def_train.csv",stringsAsFactors = TRUE)

#set a list of numerical variables
attach(dt)
#categorical = ['source_system_tab','source_type','gender','city']
#numerical = ['song_length', 'song_year']
dnum <- data.frame (song_length,song_year)
dcat <- data.frame (source_system_tab,source_type,factor(city),factor(gender),factor(language))

mca <-   MCA(dcat,ncp = 88)
summary(mca)
print(mca)
fviz_screeplot(mca, addlabels = TRUE, ylim = c(0, 45))

pc1 <- prcomp(dnum, scale=TRUE)
class(pc1)
attributes(pc1)

print(pc1)
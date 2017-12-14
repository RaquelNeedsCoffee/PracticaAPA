library(data.table)
library(FactoMineR)
library(factoextra)

# setwd("D:/FIB/PracticaAPA/Data")
# dt <- fread("df_members.csv", stringsAsFactors = TRUE)
dt <- fread("Contest/Data/samples/df_members.csv", stringsAsFactors = TRUE)

attach(dt)
dcat <- data.frame(factor(city),gender, age_range)

mca <-   MCA(dcat, ncp = 20, graph = TRUE)
# cosas podrian ser utiles en mca
# mca$eig
# mca$var$coord
# mca$var$contrib
# mca$var$v.test

eigen <- mca$eig[,1] # valores propios


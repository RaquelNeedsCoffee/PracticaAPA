library(data.table) # 
library(FactoMineR)
library(factoextra)

dt <- fread("Contest/Data/samples/definitivo.csv",stringsAsFactors = TRUE)

#set a list of numerical variables
attach(dt)

dnum <- data.frame (bd,registration_init_time,expiration_date,song_length,song_year)
dcat <- data.frame (source_system_tab,source_screen_name,source_type,factor(city),factor(registered_via),factor(gender),genre_ids,artist_name,composer,lyricist,factor(language),name,country_code,registrant_code)

mca <-   MCA(dcat,ncp = 88)
summary(mca)
print(mca)
fviz_screeplot(mca, addlabels = TRUE, ylim = c(0, 45))

pc1 <- prcomp(dnum, scale=TRUE)
class(pc1)
attributes(pc1)

print(pc1)
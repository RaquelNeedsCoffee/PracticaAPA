# Este es el data cleaning the un señor muy majo de kaggle, me falta probar que tal va

library(data.table) # 

## read data tables
members_dt <- fread("../Data/members.csv")
songs_dt <- fread("../Data/songs.csv")
train_dt <- fread("../Data/train.csv")
test_dt <- fread("../Data/test.csv")

## convert long integer to date format
standard_time <- function(i){
  # i is numeric of form 20170101
  dd<-as.character(i)
  paste0(substr(dd, 1, 4), "-", 
         substr(dd, 5, 6), "-",
         substr(dd, 7, 8))
}

members_dt[, registration_init_time :=
             as.Date(standard_time(registration_init_time))]
members_dt[, expiration_date :=
             as.Date(standard_time(expiration_date))]

## prepare combined table
train_dt [, id := -1]
test_dt [, target := -1]
both<- rbind(train_dt, test_dt)

print(sum(is.na(both)))

## Merge both with songs and members
both <- merge(both, members_dt, by = "msno", all.x=TRUE)
both <- merge(both, songs_dt, by = "song_id", all.x=TRUE)

print(sum(is.na(both)))

## Label encode the char columns
for (f in names(both)){
  if( class(both[[f]]) == "character"){
    both[is.na(both[[f]]), eval(f) := ""]
    both[, eval(f) := as.integer(
      as.factor(both[[f]]))]
  } else both[is.na(both[[f]]), eval(f) := -999]
}

## There are two date columns left
## For now to Jilian only
both[, registration_init_time := julian(registration_init_time)]
both[, expiration_date := julian(expiration_date)]
both[, length_membership := 
       expiration_date - registration_init_time]

## Make data frames ready for f.i. xgboost
setDF(both)
train_df <- both[both$id== -1,]
test_df <- both[both$target == -1,]
train_df$id <- NULL
test_df$target <- NULL

y<- train_df$target
test_id <- test_df$id
train_df$target <- NULL
test_df$id <- NULL
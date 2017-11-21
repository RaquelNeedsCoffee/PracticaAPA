library(data.table)

## read data tables
members_dt <- fread("Contest/Data/members.csv")
songs_dt <- fread("Contest/Data/songs.csv")
train_dt <- fread("Contest/Data/train.csv")
test_dt <- fread("Contest/Data/test.csv")

save(members_dt, songs_dt, train_dt, test_dt, file="Contest/Data/data_tables.RData")

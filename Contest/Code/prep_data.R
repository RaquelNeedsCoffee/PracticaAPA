library(data.table)

## read data tables
members_dt <- fread(paste(dir, "Contest/Data/members.csv", sep=""))
songs_dt <- fread(paste(dir, "Contest/Data/songs.csv", sep=""))
train_dt <- fread(paste(dir, "Contest/Data/train.csv", sep=""))
test_dt <- fread(paste(dir, "Contest/Data/test.csv", sep=""))

save(members_dt, songs_dt, train_dt, test_dt, file="Contest/Data/data_tables.RData")

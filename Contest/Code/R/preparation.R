# Preparation of the data del señor muy major de kaggle 

################################################
########## Auxiliar functions
#################################################

# au/ roc, avoid overflow error in Metrics::auc 
my_auc <- function(true_Y, probs) {
  # 
  N <- length(true_Y)
  if (length(probs) != N)
    return (NULL) # error
  if (is.factor(true_Y)) true_Y <- as.numeric(as.character(true_Y))
  roc_y <- true_Y[order(probs, decreasing = FALSE)]
  stack_x = cumsum(roc_y == 1) / sum(roc_y == 1)
  stack_y = cumsum(roc_y == 0) / sum(roc_y == 0)
  auc = sum((stack_x[2:N] - stack_x[1:(N - 1)]) * stack_y[2:N])
  return(auc)
}

auc <- function(a,p) my_auc(a,p)

#binary cross_entropy
bce <- function(actual, probs){
  probs <- ifelse(probs >0, probs, 10^-10)
  return ( - mean(actual* log(probs)))
}
# mean logloss
mll <- function(actual, probs){
  probs <- ifelse(probs >0, probs, 10^-10)
  return ( mean(Metrics::ll(actual, probs)))
}

# accuracy
acc <- function(actual, probs, theta=0.5){
  probs <- ifelse(probs > theta, 1, 0)
  return(mean(probs == actual))
}

# root mean squared error
rmse <- function(actuals, prediction) sqrt(mean((actuals-prediction)^2))

diagnosis <- function(actual, probs, title=""){
  cat("\nSummary results for", title
      , "\nauc:", auc(actual, probs)
      , "\n Accuracy:", acc(actual, probs)
      , "\n Binary cross entropy:", bce(actual, probs)
      , "\n Mean Logloss:", mll(actual, probs)
      , "\n Root mean squared error:", rmse(actual, probs)
      , "\n"
  )
}

# primitive (0,1) calibration
to_p <- function(r) {
  r <- r - min(r)
  return(r / max(r))
}

####################################################
################ Split data ######################## 
####################################################

# ---------------------------------------------
set.seed(3141569)
E_SIZE <- 0.185
h<- sample(nrow(train_df), E_SIZE*nrow(train_df))
ens_val <- train_df[h, ]
y_ens_val <- y[h]
train_df <- train_df[-h, ]
y <- y[-h]

# now only _after_ this split we can scale the columns
to_do <- names(train_df)
for (f in to_do){
  mm<- mean(train_df[[f]])
  ss<- sd(train_df[[f]])
  train_df[[f]] <- (train_df[[f]] -mm)/ss
  test_df[[f]] <- (test_df[[f]] -mm)/ss
  ens_val[[f]] <- (ens_val[[f]] -mm)/ss
}

# shrink? Then use this subset
subs <- sample(nrow(train_df), 1*nrow(train_df))



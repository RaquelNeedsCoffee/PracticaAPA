# Random forest del señor majo

#######################################
####### Random forest ################
######################################
library(ranger)
rf <- ranger(y[subs] ~ . , data = train_df[subs,], num.trees = 12
             , verbose= FALSE)

pred_1_e<-predict(rf, ens_val, type = "response")
pred_1_e <- pred_1_e$predictions
pred_1_t<-predict(rf, test_df, type = "response")
pred_1_t <- pred_1_t$predictions

diagnosis(y_ens_val, pred_1_e, title="ranger")

#######################################
####### Neural net ussing keras #######
#######################################

B_Size= 2^15
x_train<- as.matrix(train_df)
y_train<- as.matrix(data.frame(p=1-y, q=y))

model <- keras_model_sequential()
model %>%
  layer_dense(
    units= 128,
    input_shape = c(ncol(x_train)),
    kernel_initializer='he_normal' #,
  ) %>%
  layer_activation("relu") %>%
  layer_batch_normalization() %>%
  layer_dropout(rate= 0.1) %>%
  
  layer_dense(
    units= 512,
    kernel_initializer='he_normal' #,
  ) %>%
  layer_activation("relu") %>%
  layer_batch_normalization() %>%
  layer_dropout(rate= 0.3) %>%
  
  layer_dense(
    units= 64,
    kernel_initializer='he_normal' #,
  ) %>%
  layer_activation("relu") %>%
  layer_batch_normalization() %>%
  layer_dropout(rate= 0.1) %>%
  
  layer_dense(2) %>%
  layer_activation("softmax")

model %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = optimizer_rmsprop(),
  metrics = c("accuracy")
)

history <- model %>% fit(
  x_train, y_train, verbose=2, 
  view_metrics = FALSE, 
  epochs = 7, batch_size =B_Size, 
  validation_split = 0.1
)

pred_2_e <- model %>% predict(as.matrix(ens_val[, to_do])
                              , batch_size = B_Size)
pred_2_t <- model %>% predict(as.matrix(test_df[, to_do])
                              , batch_size = B_Size)
pred_2_e <- pred_2_e [,2]
pred_2_t <- pred_2_t [,2]
diagnosis(y_ens_val, pred_2_e
          , title="keras")

#######################################
############# xgboots  ################
#######################################
library(xgboost)
param = list(
  objective="binary:logistic",
  eval_metric= "auc",
  subsample= 0.95,
  colsample_bytree=0.45, 
  max_depth= 10,
  min_child= 6,
  tree_method= "approx", 
  eta  = 0.9 , 
  nthreads = 8
)
x_train <- xgb.DMatrix(
  as.matrix(train_df[subs,]),
  label = y[subs], 
  missing=-1)
x_val <- xgb.DMatrix(
  as.matrix(ens_val), 
  label = y_ens_val, missing=-1)
x_test <- xgb.DMatrix(as.matrix(test_df), missing= -1)

model <- xgb.train(
  data = x_train,
  nrounds = 100, 
  params = param,  
  maximize= T,
  watchlist = list(val = x_val),
  print_every_n = 50
)

pred_3_e  <- predict(model, x_val)
pred_3_t  <- predict(model, x_test) 
diagnosis(y_ens_val, pred_3_e, title="xgb")


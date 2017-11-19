#########################################################################
######################### Rate the ensamble #############################
#########################################################################

cat("\n=== Optimizing for ensemble start:",  format(Sys.time()), "===\n")

e_m<- cbind(V1= to_p(pred_1_e),
            V2= to_p(pred_2_e),
            V3= to_p(pred_3_e))

t_m<- cbind(V1= to_p(pred_1_t),
            V2= to_p(pred_2_t),
            V3= to_p(pred_3_t))

m_fun <- function(par){
  return(1-auc(y_ens_val, (e_m %*% par)/sum(par)))
}
start_par=rep(1, ncol(e_m))/ ncol(e_m)
a <- optim(start_par, m_fun, method= "L-BFGS-B",
           lower = 0, upper = 1)
#
submit_prediction <- (t_m %*% a$par)/sum(a$par)
#
diagnosis(y_ens_val, (e_m %*% a$par)/sum(a$par), title="optimizer"); cat(
  "\n=== Ensembling end:",  format(Sys.time()), 
  "===\n=== with parameters", unlist(round(a$par/sum(a$par),3)), "\n")


require(readr)

subm <- read_csv("../input/sample_submission.csv")

subm$id<- as.integer(subm$id)
subm[, 2] <- round(submit_prediction, 3)
write.csv(subm, "mrooijers001.csv", row.names = F) 

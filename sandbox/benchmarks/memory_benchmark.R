rm(list=ls())
library(pryr)
library(data.table)
library(SuperLearner)
#make big cpp data

<<<<<<< HEAD
data(cpp)
cpp <- cpp[!is.na(cpp[, "haz"]), ]
covars <- c("apgar1", "apgar5", "parity", "gagebrth", "mage", "meducyrs", "sexn")
cpp[is.na(cpp)] <- 0
cpp <- cpp[sample(nrow(cpp),50000,replace=T),]
cpp$haz_bin <- cpp$haz > median(cpp$haz)
outcome="haz_bin"

folds <- origami::make_folds(cpp)

#define learners
lrn_mean <- sl3::Lrnr_mean$new()
=======

>>>>>>> 99b29ec96037f2bd76d2b0e103c94fe892e0644d
lrn_glm_all <<- sl3::Lrnr_glm_fast$new(family = "binomial")
lrn_gbm <<-  sl3::Lrnr_xgboost$new(objective = 'reg:logistic',
                                   booster = "gbtree",
                                   nrounds = 100,
                                   learning_rate = .1,
                                   nthread = 1,
                                   max_delta_step = 5,
                                   max_depth = 3,
                                   gamma = .5,
                                   colsample_bytree = .5,
                                   subsample = .8,
                                   lambda = 4,
                                   alpha = 1)

glmnet_learner <- sl3::Lrnr_pkg_SuperLearner$new("SL.glmnet", family = "gaussian")

<<<<<<< HEAD
sl <- sl3::Lrnr_sl$new(learners = sl3::Stack$new(lrn_gbm, lrn_glm_all, glmnet_learner),
                       metalearner = sl3::Lrnr_nnls$new(), folds=folds)


make_big_cpp_task <- function(cpp){
  cpp_copy <- copy(cpp)
  task <- sl3_Task$new(cpp_copy, covariates = covars, outcome = outcome, folds=NULL)
=======
sl <- sl3::Lrnr_sl$new(learners = sl3::Stack$new(lrn_glm_all, lrn_gbm, glmnet_learner),
                       metalearner = sl3::Lrnr_nnls$new())


data(cpp)
cpp <- cpp[!is.na(cpp[, "haz"]), ]
covars <- c("apgar1", "apgar5", "parity", "gagebrth", "mage", "meducyrs", "sexn")
cpp[is.na(cpp)] <- 0
cpp <- cpp[sample(nrow(cpp),10000,replace=T),]
cpp$haz_bin <- cpp$haz > median(cpp$haz)
outcome="haz_bin"

make_big_cpp_task <- function(cpp){
  cpp_copy <- copy(cpp)
  task <- sl3_Task$new(cpp_copy, covariates = covars, outcome = outcome)
>>>>>>> 99b29ec96037f2bd76d2b0e103c94fe892e0644d
  
  return(task)
}

starting_mem <- mem_used()
define_task <- mem_change({new_task <- make_big_cpp_task(cpp)})
rm_task <- mem_change({rm(new_task)})

redefine_task <- mem_change({new_task <- make_big_cpp_task(cpp)})
task_object_size <- object_size(new_task)

<<<<<<< HEAD
folds <- new_task$folds
new_task$.__enclos_env__$private$.folds=NULL

fit_sl <- mem_change({sl_fit <- sl$train(new_task)})
fit_object_size <- object_size(sl_fit)
gc()
mem_after_fit <- mem_used()


internal_ref <- new_task$data
naive_rm <- mem_change({rm(new_task)})
=======
fit_sl <- mem_change({sl_fit <- sl$train(new_task)})
fit_object_size <- object_size(sl_fit)
mem_after_fit <- mem_used()

naive_rm <- mem_change({rm(new_task)})
internal_ref <- sl_fit$training_task$data
>>>>>>> 99b29ec96037f2bd76d2b0e103c94fe892e0644d
force_rm <- mem_change({set(internal_ref, j=names(internal_ref), value=NULL)})

fit_after_rm <- object_size(sl_fit)

<<<<<<< HEAD
gc()
mem_after_rm <- mem_used()
total_fit <- mem_after_fit-starting_mem
total_fit_after_rm <- mem_after_rm-starting_mem
sum(sapply(sl_fit$fit_object,object_size))/2^20
object_size(sl_fit$fit_object)
=======
mem_after_rm <- mem_used()
total_fit <- mem_after_fit-starting_mem
total_fit_after_rm <- mem_after_rm-starting_mem

>>>>>>> 99b29ec96037f2bd76d2b0e103c94fe892e0644d
sizes <- c(define_task=define_task,
           rm_task=rm_task,
           redefine_task=redefine_task,
           fit_sl=fit_sl,
           naive_rm=naive_rm,
           force_rm=force_rm,
           
           fit=fit_object_size,
           task=task_object_size,
           fit_after_rm=fit_after_rm,
           
           total_after_fit=total_fit,
           total_after_rm=total_fit_after_rm)
print(sizes/2^20)
<<<<<<< HEAD

=======
>>>>>>> 99b29ec96037f2bd76d2b0e103c94fe892e0644d

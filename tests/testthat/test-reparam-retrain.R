context("test_reparameterize-retrain.R -- Learner reparameterization & retraining")

data(cpp_imputed)
covars <- c("apgar1", "apgar5", "parity", "gagebrth", "mage", "meducyrs", "sexn")
outcome <- "haz"
task <- sl3_Task$new(data.table::copy(cpp_imputed), covariates = covars, outcome = outcome)
glm_lrnr <- Lrnr_glm$new()
original_fit <- glm_lrnr$train(task)

test_that("We can reparameterize an untrained model", {
  new_params <- list(covariates = setdiff(covars, "sexn"))
  reparam_lrnr_from_lrnr <- glm_lrnr$reparameterize(new_params)
  reparam_fit_from_lrnr <- reparam_lrnr_from_lrnr$train(task)
  expect_true(reparam_fit_from_lrnr$is_trained)
  expect_equal(setdiff(names(coef(original_fit)), names(coef(reparam_fit_from_lrnr))), "sexn")
})

test_that("We can reparameterize a trained model and refit", {
  new_params <- list(covariates = setdiff(covars, "sexn"))
  reparam_lrnr_from_fit <- original_fit$reparameterize(new_params)
  reparam_fit_from_fit <- reparam_lrnr_from_fit$train(task)
  expect_true(reparam_fit_from_fit$is_trained)
  expect_equal(setdiff(names(coef(original_fit)), names(coef(reparam_fit_from_fit))), "sexn")
})

test_that("We cannot retrain a model on a new task with train", {
  new_covars_task <- sl3_Task$new(data.table::copy(cpp_imputed),
    covariates = covars[-7], outcome = outcome
  )
  expect_error(original_fit$train(new_covars_task))
})

test_that("We can retrain a model on a new task with new covariates", {
  new_covars_task <- sl3_Task$new(data.table::copy(cpp_imputed),
    covariates = covars[-7], outcome = outcome
  )
  retrained_fit <- original_fit$retrain(new_covars_task)
  expect_true(retrained_fit$is_trained)
  expect_equal(setdiff(names(coef(original_fit)), names(coef(retrained_fit))), "sexn")
})

test_that("We can retrain a model on a new task with new covariates and outcome", {
  new_outcome_task <- sl3_Task$new(data.table::copy(cpp_imputed),
    covariates = covars[-7], outcome = "sexn"
  )
  retrained_fit <- original_fit$retrain(new_outcome_task)
  expect_equal(retrained_fit$training_task$outcome_type$type, "binomial")
})

#' Coefficient Magnitude Screener
#'
#' This learner provides screening of covariates based on the magnitude of
#' their estimated coefficients in a (possibly regularized) GLM.
#'
#' @docType class
#'
#' @importFrom R6 R6Class
#'
#' @export
#'
#' @keywords data
#'
#' @return Learner object with methods for training and prediction. See
#'  \code{\link{Lrnr_base}} for documentation on learners.
#'
#' @format \code{\link{R6Class}} object.
#'
#' @family Learners
#'
#' @section Parameters:
#' \describe{
#'   \item{\code{learner}}{An instantiated learner to use for estimating
#'     coefficients used in screening.}
#'   \item{\code{threshold = 1e-3}}{Minimum size of coefficients to be kept.}
#'   \item{\code{max_retain = NULL}}{Maximum no. variables to be kept.}
#'   \item{\code{...}}{Other parameters passed to \code{learner}.}
#' }
Lrnr_screener_coefs <- R6Class(
  classname = "Lrnr_screener_coefs",
  inherit = Lrnr_base, portable = TRUE, class = TRUE,
  public = list(
    initialize = function(learner, threshold = 1e-3, max_retain = NULL, ...) {
      params <- args_to_list()
      super$initialize(params = params, ...)
    }
  ),
  private = list(
    .properties = c("screener"),

    .train = function(task) {
      learner <- self$params$learner
      fit <- learner$train(task)
      coefs <- as.vector(coef(fit))
      coef_names <- rownames(coef(fit))
      if (is.null(coef_names)) {
        coef_names <- names(coef(fit))
      }

      if (is.null(coef_names)) {
        stop("could not extract names from fit coefficients")
      }

      covs <- task$nodes$covariates

      selected_coefs <- coef_names[which(abs(coefs) > self$params$threshold)]
      selected_coefs <- unique(gsub("\\..*", "", selected_coefs))
      selected <- intersect(selected_coefs, covs)

      if (!is.null(self$params$max_retain) &&
        (self$params$max_retain < length(selected))) {
        ord_coefs <- coef_names[order(abs(coefs), decreasing = TRUE)]
        ord_coefs <- unique(gsub("\\..*", "", ord_coefs))
        selected <- intersect(ord_coefs, covs)[1:self$params$max_retain]
      }

      fit_object <- list(selected = selected)
      return(fit_object)
    },

    .predict = function(task) {
      task$X[, private$.fit_object$selected, with = FALSE, drop = FALSE]
    },

    .chain = function(task) {
      return(task$next_in_chain(covariates = private$.fit_object$selected))
    },
    .required_packages = c()
  )
)

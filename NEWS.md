# sl3 1.3.1
* Overhaul of data preprocessing.
* New screening methods and convex combination in `Lrnr_nnls`.
* Bug fixes including covariate subsetting and better handling of NAs.
* Package and documentation cleanup.

# sl3 1.3.0
* Fixes incorrect handling of missingness in the automatic imputation procedure.
* Adds new standard learners, including from the `gam` and `caret` packages.
* Adds custom learners for conditional density estimation, including
  semiparametric methods based on conditional mean and conditional mean/variance
  estimation as well as generalized functionality for density estimation via a
  pooled hazards approach.

# sl3 1.2.0
* Default metalearners based on task outcome types.
* Handling of imputation internally in task objects.
* Addition of several new learners, including from the `gbm`, `earth`,
  `polspline` packages.
* Fixing errors in existing learners (e.g., subtle parallelization in `xgboost`
  and `ranger`).
* Support for multivariate outcomes
* Sets default cross-validation to be revere-style.
* Support for cross-validated super learner and variable importance.

# sl3 1.1.0
* A full-featured and stable release of the project.
* Numerous learners are included and many bugs have been fixed relative to
  earlier versions (esp v1.0.0) of the software.

# sl3 1.0.0
* An initial stable release.

% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/metalearners.R
\name{metalearners}
\alias{metalearners}
\alias{metalearner_logistic_binomial}
\alias{metalearner_linear}
\alias{metalearner_linear_multivariate}
\alias{metalearner_linear_multinomial}
\title{Combine predictions from multiple learners}
\usage{
metalearner_logistic_binomial(alpha, X, trim)

metalearner_linear(alpha, X)

metalearner_linear_multivariate(alpha, X)

metalearner_linear_multinomial(alpha, X)
}
\arguments{
\item{alpha}{a vector of combination coefficients}

\item{X}{a matrix of predictions}

\item{trim}{a value use to trim predictions away from 0 and 1.}
}
\description{
Combine predictions from multiple learners
}

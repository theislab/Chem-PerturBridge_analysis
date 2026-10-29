#!/usr/bin/env Rscript
# Adaptive shrinkage posterior means for one group of effect estimates.
#
# Reads a raw little-endian float64 stream of 2n values (n betahat followed by n
# standard errors) and writes n float64 posterior means in the same order.
#
# The prior g is fitted once on a subsample and then held fixed while the
# posterior pass runs in chunks. Holding g fixed is what guarantees that every
# estimate in the group is shrunk through the identical prior, which is the
# property the retrieval comparison depends on; chunking keeps the n x K
# likelihood matrix from being materialised all at once.

suppressPackageStartupMessages(library(ashr))

parse_args <- function(argv) {
  out <- list(subsample = 2e5L, seed = 0L, chunk = 2e6L, mixcompdist = "normal",
              fit_input = NULL, fit_n = 0L)
  i <- 1L
  while (i <= length(argv)) {
    key <- argv[[i]]
    value <- if (i < length(argv)) argv[[i + 1L]] else NA_character_
    if (key == "--input") out$input <- value
    else if (key == "--output") out$output <- value
    else if (key == "--n") out$n <- as.integer(value)
    else if (key == "--fit-input") out$fit_input <- value
    else if (key == "--fit-n") out$fit_n <- as.integer(value)
    else if (key == "--subsample") out$subsample <- as.integer(value)
    else if (key == "--seed") out$seed <- as.integer(value)
    else if (key == "--chunk") out$chunk <- as.integer(value)
    else if (key == "--mixcompdist") out$mixcompdist <- value
    else stop("Unknown argument: ", key)
    i <- i + 2L
  }
  for (required in c("input", "output", "n")) {
    if (is.null(out[[required]])) stop("Missing required argument: --", required)
  }
  out
}

read_pairs <- function(path, n) {
  con <- file(path, "rb")
  values <- readBin(con, what = "double", n = 2L * n, size = 8L, endian = "little")
  close(con)
  if (length(values) != 2L * n) {
    stop(sprintf("Expected %d doubles in %s, read %d", 2L * n, path, length(values)))
  }
  list(betahat = values[seq_len(n)], sebetahat = values[(n + 1L):(2L * n)])
}

args <- parse_args(commandArgs(trailingOnly = TRUE))
n <- args$n

main_data <- read_pairs(args$input, n)
betahat <- main_data$betahat
sebetahat <- main_data$sebetahat
rm(main_data)

# The prior is fitted on its own pool when one is supplied, which lets the
# caller restrict the fit to a cleaner subset (for instance genes tested in
# every batch) while still shrinking every estimate.
if (!is.null(args$fit_input)) {
  fit_data <- read_pairs(args$fit_input, args$fit_n)
  fit_betahat <- fit_data$betahat
  fit_sebetahat <- fit_data$sebetahat
  rm(fit_data)
} else {
  fit_betahat <- betahat
  fit_sebetahat <- sebetahat
}

# ash only needs enough draws to pin down the mixture weights, and every group
# here has far more than that.
set.seed(args$seed)
n_fit_pool <- length(fit_betahat)
fit_idx <- if (n_fit_pool > args$subsample) sample.int(n_fit_pool, args$subsample) else seq_len(n_fit_pool)
prior_fit <- ash(
  fit_betahat[fit_idx],
  fit_sebetahat[fit_idx],
  mixcompdist = args$mixcompdist,
  outputlevel = 0
)
g <- get_fitted_g(prior_fit)
rm(prior_fit, fit_betahat, fit_sebetahat)

posterior <- numeric(n)
starts <- seq.int(1L, n, by = args$chunk)
for (start in starts) {
  stop_at <- min(start + args$chunk - 1L, n)
  idx <- start:stop_at
  chunk_fit <- ash(
    betahat[idx],
    sebetahat[idx],
    g = g,
    fixg = TRUE,
    mixcompdist = args$mixcompdist
  )
  posterior[idx] <- get_pm(chunk_fit)
}

con <- file(args$output, "wb")
writeBin(posterior, con, size = 8L, endian = "little")
close(con)

cat(sprintf(
  "ashr: n=%d fitted_on=%d components=%d\n",
  n, length(fit_idx), length(g$pi)
))

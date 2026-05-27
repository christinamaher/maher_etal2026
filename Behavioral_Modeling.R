# General selective-attention and uniform-attention reinforcement learning models
#
# This file contains core functions for fitting two feature-based RL models:
#
# 1. Uniform Attention RL (UA-RL)
#    Feature dimensions are weighted equally when computing option values and
#    updating feature values.
#
# 2. Selective Attention RL (SA-RL) 
#    A free attention parameter, phi, weights the currently relevant feature
#    dimension more strongly than irrelevant dimensions.
#

# Expected dataframe structure
# ----------------------------
# One row per trial. At minimum, the dataframe should contain:
#
# - block column: block/game/run identifier
# - choice column: integer index of chosen option on that trial, e.g. 1, 2, 3
# - reward column: numeric reward/outcome
# - relevant-dimension column: dimension to emphasize for SA-RL/ACL
# - option feature columns: one column per option per dimension
#
# Example option_cols for a 3-option task with 2 feature dimensions:
#
# option_cols <- list(
#   color = c("option1_color", "option2_color", "option3_color"),
#   shape = c("option1_shape", "option2_shape", "option3_shape")
# )
#
# The relevant-dimension column should contain values matching names(option_cols),
# e.g. "color" or "shape".


# -----------------------------------------------------------------------------
# Basic utilities
# -----------------------------------------------------------------------------

inv_logit <- function(x) {
  #' Inverse-logit transform.
  #'
  #' @param x Numeric value or vector.
  #' @return Value transformed to the interval (0, 1).
  1 / (1 + exp(-x))
}


safe_softmax <- function(x, beta) {
  #' Numerically stable softmax.
  #'
  #' @param x Numeric vector of option values.
  #' @param beta Inverse-temperature parameter.
  #' @return Vector of choice probabilities summing to 1.
  z <- beta * x
  z <- z - max(z, na.rm = TRUE)
  exp_z <- exp(z)
  exp_z / sum(exp_z, na.rm = TRUE)
}


initialize_feature_values <- function(data, option_cols, initial_value = NULL) {
  #' Initialize feature values for each dimension.
  #'
  #' @param data Trial-level dataframe.
  #' @param option_cols Named list mapping dimensions to option-feature columns.
  #' @param initial_value Optional numeric initial value assigned to every feature.
  #'   If NULL, defaults to 1 / number of unique feature identities.
  #'
  #' @return Nested named list of feature values, one vector per dimension.
  feature_values <- list()

  all_features <- unique(unlist(data[unlist(option_cols)]))
  all_features <- all_features[!is.na(all_features)]

  if (is.null(initial_value)) {
    initial_value <- 1 / length(all_features)
  }

  for (dimension in names(option_cols)) {
    features <- unique(unlist(data[option_cols[[dimension]]]))
    features <- features[!is.na(features)]
    feature_values[[dimension]] <- stats::setNames(
      rep(initial_value, length(features)),
      as.character(features)
    )
  }

  feature_values
}


get_attention_weights <- function(
  model,
  dimensions,
  relevant_dimension = NULL,
  phi = NULL
) {
  #' Compute dimension-level attention weights.
  #'
  #' @param model Either "UA" or "SA".
  #' @param dimensions Character vector of feature-dimension names.
  #' @param relevant_dimension Relevant dimension for the current block/trial.
  #' @param phi Selective-attention weight used by SA-RL/ACL.
  #'
  #' @return Named numeric vector of attention weights.
  model <- match.arg(model, c("UA", "SA"))
  n_dims <- length(dimensions)

  if (model == "UA") {
    return(stats::setNames(rep(1 / n_dims, n_dims), dimensions))
  }

  if (is.null(relevant_dimension)) {
    stop("SA-RL requires relevant_dimension.", call. = FALSE)
  }
  if (!relevant_dimension %in% dimensions) {
    stop("relevant_dimension must match one of names(option_cols).", call. = FALSE)
  }
  if (is.null(phi)) {
    stop("SA-RL requires phi.", call. = FALSE)
  }
  if (phi < 0 || phi > 1) {
    stop("phi must be between 0 and 1.", call. = FALSE)
  }

  weights <- rep((1 - phi) / (n_dims - 1), n_dims)
  names(weights) <- dimensions
  weights[relevant_dimension] <- phi
  weights
}


get_option_features <- function(trial_row, option_cols) {
  #' Extract option features for one trial.
  #'
  #' @param trial_row One-row dataframe.
  #' @param option_cols Named list mapping dimensions to option-feature columns.
  #'
  #' @return Matrix/dataframe with rows = options and columns = dimensions.
  dimensions <- names(option_cols)
  n_options <- length(option_cols[[1]])

  option_features <- data.frame(matrix(NA_character_, nrow = n_options, ncol = length(dimensions)))
  colnames(option_features) <- dimensions

  for (dimension in dimensions) {
    option_features[[dimension]] <- as.character(unlist(trial_row[option_cols[[dimension]]]))
  }

  option_features
}


compute_option_values <- function(option_features, feature_values, attention_weights) {
  #' Compute attention-weighted option values.
  #'
  #' @param option_features Dataframe with rows = options, columns = dimensions.
  #' @param feature_values Nested list of learned feature values.
  #' @param attention_weights Named numeric vector of dimension weights.
  #'
  #' @return Numeric vector of option values.
  values <- numeric(nrow(option_features))

  for (dimension in names(attention_weights)) {
    features <- option_features[[dimension]]
    dim_values <- feature_values[[dimension]][features]
    values <- values + attention_weights[[dimension]] * as.numeric(dim_values)
  }

  values
}


update_chosen_feature_values <- function(
  feature_values,
  chosen_features,
  prediction_error,
  alpha,
  attention_weights
) {
  #' Update values for the chosen option's features.
  #'
  #' @param feature_values Nested list of learned feature values.
  #' @param chosen_features Named vector of features for the chosen option.
  #' @param prediction_error Reward prediction error.
  #' @param alpha Learning-rate parameter.
  #' @param attention_weights Named numeric vector of dimension weights.
  #'
  #' @return Updated feature_values list.
  for (dimension in names(attention_weights)) {
    feature <- as.character(chosen_features[[dimension]])
    feature_values[[dimension]][feature] <-
      feature_values[[dimension]][feature] +
      alpha * prediction_error * attention_weights[[dimension]]
  }

  feature_values
}


# -----------------------------------------------------------------------------
# Core model likelihoods
# -----------------------------------------------------------------------------

run_feature_rl_model <- function(
  params,
  data,
  option_cols,
  model = c("UA", "SA"),
  block_col = "block",
  choice_col = "choice",
  reward_col = "reward",
  relevant_dimension_col = "relevant_dimension",
  transform_phi = TRUE,
  return_trialwise = FALSE,
  initial_value = NULL
) {
  #' Run a feature-based RL model and compute negative log likelihood.
  #'
  #' @param params Numeric vector of model parameters.
  #'   For UA-RL: c(alpha, beta).
  #'   For SA-RL/ACL: c(alpha, phi, beta), where phi may be unconstrained if
  #'   transform_phi = TRUE.
  #' @param data Trial-level dataframe.
  #' @param option_cols Named list mapping feature dimensions to option columns.
  #' @param model "UA" for uniform-attention RL or "SA" for selective-attention RL.
  #' @param block_col Column identifying blocks/games/runs.
  #' @param choice_col Column giving chosen option index on each trial.
  #' @param reward_col Column giving observed reward/outcome on each trial.
  #' @param relevant_dimension_col Column giving relevant dimension for SA-RL.
  #' @param transform_phi Logical. If TRUE, applies inverse-logit to params[2]
  #'   for SA-RL so unconstrained optimization can be used.
  #' @param return_trialwise Logical. If TRUE, returns trial-level predictions
  #'   instead of only negative log likelihood.
  #' @param initial_value Optional initial value for all feature values.
  #'
  #' @return Negative log likelihood, or a list with negative log likelihood and
  #'   trialwise predictions if return_trialwise = TRUE.
  model <- match.arg(model)
  dimensions <- names(option_cols)

  alpha <- params[1]

  if (model == "UA") {
    beta <- params[2]
    phi <- NA_real_
  } else {
    phi <- if (transform_phi) inv_logit(params[2]) else params[2]
    beta <- params[3]
  }

  if (alpha < 0 || alpha > 1 || beta < 0) {
    return(Inf)
  }
  if (model == "SA" && (phi < 0 || phi > 1)) {
    return(Inf)
  }

  data <- data[order(data[[block_col]]), , drop = FALSE]
  blocks <- unique(data[[block_col]])

  choice_probs <- numeric(0)
  trialwise <- list()
  row_idx <- 1

  for (block in blocks) {
    block_data <- data[data[[block_col]] == block, , drop = FALSE]
    feature_values <- initialize_feature_values(
      block_data,
      option_cols = option_cols,
      initial_value = initial_value
    )

    for (trial in seq_len(nrow(block_data))) {
      trial_row <- block_data[trial, , drop = FALSE]
      option_features <- get_option_features(trial_row, option_cols)

      relevant_dimension <- if (model == "SA") {
        as.character(trial_row[[relevant_dimension_col]])
      } else {
        NULL
      }

      attention_weights <- get_attention_weights(
        model = model,
        dimensions = dimensions,
        relevant_dimension = relevant_dimension,
        phi = phi
      )

      option_values <- compute_option_values(
        option_features = option_features,
        feature_values = feature_values,
        attention_weights = attention_weights
      )

      probabilities <- safe_softmax(option_values, beta = beta)

      choice <- as.integer(trial_row[[choice_col]])
      reward <- as.numeric(trial_row[[reward_col]])

      if (is.na(choice) || choice < 1 || choice > length(probabilities)) {
        stop("Choice values must be integer option indices from 1 to n_options.", call. = FALSE)
      }

      chosen_prob <- max(probabilities[choice], .Machine$double.eps)
      choice_probs <- c(choice_probs, chosen_prob)

      prediction_error <- reward - option_values[choice]
      chosen_features <- option_features[choice, , drop = TRUE]

      feature_values <- update_chosen_feature_values(
        feature_values = feature_values,
        chosen_features = chosen_features,
        prediction_error = prediction_error,
        alpha = alpha,
        attention_weights = attention_weights
      )

      if (return_trialwise) {
        trialwise[[row_idx]] <- data.frame(
          block = block,
          trial_in_block = trial,
          choice = choice,
          reward = reward,
          choice_probability = chosen_prob,
          expected_value = option_values[choice],
          prediction_error = prediction_error
        )
        row_idx <- row_idx + 1
      }
    }
  }

  neg_log_likelihood <- -sum(log(choice_probs))

  if (return_trialwise) {
    return(list(
      neg_log_likelihood = neg_log_likelihood,
      trialwise = do.call(rbind, trialwise)
    ))
  }

  neg_log_likelihood
}


choice_likelihood <- function(
  params,
  data,
  option_cols,
  model = c("UA", "SA"),
  ...
) {
  #' Compute average held-out choice likelihood.
  #'
  #' This mirrors the held-out likelihood calculation used after fitting on
  #' training blocks.
  #'
  #' @param params Model parameters.
  #' @param data Held-out trial-level dataframe.
  #' @param option_cols Named list mapping feature dimensions to option columns.
  #' @param model "UA" or "SA".
  #' @param ... Additional arguments passed to run_feature_rl_model().
  #'
  #' @return Mean probability assigned to the observed choices.
  out <- run_feature_rl_model(
    params = params,
    data = data,
    option_cols = option_cols,
    model = model,
    return_trialwise = TRUE,
    ...
  )

  mean(out$trialwise$choice_probability, na.rm = TRUE)
}


# -----------------------------------------------------------------------------
# Model fitting
# -----------------------------------------------------------------------------

fit_feature_rl_model <- function(
  data,
  option_cols,
  model = c("UA", "SA"),
  start_params = NULL,
  block_col = "block",
  choice_col = "choice",
  reward_col = "reward",
  relevant_dimension_col = "relevant_dimension",
  initial_value = NULL,
  method = "L-BFGS-B"
) {
  #' Fit a feature-based RL model by minimizing negative log likelihood.
  #'
  #' @param data Trial-level dataframe.
  #' @param option_cols Named list mapping feature dimensions to option columns.
  #' @param model "UA" or "SA".
  #' @param start_params Optional starting parameters.
  #'   UA-RL: c(alpha, beta). SA-RL: c(alpha, phi_unconstrained, beta).
  #' @param block_col Column identifying blocks/games/runs.
  #' @param choice_col Column giving chosen option index.
  #' @param reward_col Column giving reward/outcome.
  #' @param relevant_dimension_col Column giving relevant dimension for SA-RL.
  #' @param initial_value Optional initial feature value.
  #' @param method Optimization method.
  #'
  #' @return Dataframe with BIC, fitted parameters, and negative log likelihood.
  model <- match.arg(model)

  if (is.null(start_params)) {
    if (model == "UA") {
      start_params <- c(runif(1, 0, 1), rgamma(1, 2, 3))
    } else {
      start_params <- c(runif(1, 0, 1), stats::qlogis(runif(1, 0.5, 1)), rgamma(1, 2, 3))
    }
  }

  lower <- if (model == "UA") c(0, 0) else c(0, -Inf, 0)
  upper <- if (model == "UA") c(1, Inf) else c(1, Inf, Inf)

  fit <- stats::optim(
    par = start_params,
    fn = run_feature_rl_model,
    data = data,
    option_cols = option_cols,
    model = model,
    block_col = block_col,
    choice_col = choice_col,
    reward_col = reward_col,
    relevant_dimension_col = relevant_dimension_col,
    initial_value = initial_value,
    method = method,
    lower = lower,
    upper = upper
  )

  neg_log_likelihood <- fit$value
  bic <- length(start_params) * log(nrow(data)) + 2 * neg_log_likelihood

  if (model == "UA") {
    out <- data.frame(
      model = "UA",
      bic = bic,
      learning_rate = fit$par[1],
      phi = NA_real_,
      beta = fit$par[2],
      neg_log_likelihood = neg_log_likelihood,
      convergence = fit$convergence
    )
  } else {
    out <- data.frame(
      model = "SA",
      bic = bic,
      learning_rate = fit$par[1],
      phi = inv_logit(fit$par[2]),
      phi_unconstrained = fit$par[2],
      beta = fit$par[3],
      neg_log_likelihood = neg_log_likelihood,
      convergence = fit$convergence
    )
  }

  out
}


fit_models_leave_one_block_out <- function(
  data,
  option_cols,
  n_iter = 5,
  block_col = "block",
  choice_col = "choice",
  reward_col = "reward",
  relevant_dimension_col = "relevant_dimension",
  initial_value = NULL,
  seed = NULL
) {
  #' Fit UA-RL and SA-RL/ACL using leave-one-block-out cross-validation.
  #'
  #' For each random initialization and each held-out block, the models are fit
  #' on all other blocks and evaluated by average choice likelihood on the
  #' held-out block.
  #'
  #' @param data Trial-level dataframe.
  #' @param option_cols Named list mapping feature dimensions to option columns.
  #' @param n_iter Number of random starting-point iterations.
  #' @param block_col Column identifying blocks/games/runs.
  #' @param choice_col Column giving chosen option index.
  #' @param reward_col Column giving reward/outcome.
  #' @param relevant_dimension_col Column giving relevant dimension for SA-RL.
  #' @param initial_value Optional initial feature value.
  #' @param seed Optional random seed.
  #'
  #' @return Named list with `ua`, `sa`, and `starting_params` dataframes.
  if (!is.null(seed)) {
    set.seed(seed)
  }

  blocks <- unique(data[[block_col]])
  ua_results <- list()
  sa_results <- list()
  starting_rows <- list()

  row_counter <- 1

  for (iter in seq_len(n_iter)) {
    start_alpha <- runif(1, 0, 1)
    start_phi <- stats::qlogis(runif(1, 0.5, 1))
    start_beta <- rgamma(1, 2, 3)

    starting_rows[[iter]] <- data.frame(
      iteration = iter,
      alpha = start_alpha,
      phi_unconstrained = start_phi,
      beta = start_beta
    )

    for (heldout_block in blocks) {
      train_data <- data[data[[block_col]] != heldout_block, , drop = FALSE]
      test_data <- data[data[[block_col]] == heldout_block, , drop = FALSE]

      ua_fit <- fit_feature_rl_model(
        data = train_data,
        option_cols = option_cols,
        model = "UA",
        start_params = c(start_alpha, start_beta),
        block_col = block_col,
        choice_col = choice_col,
        reward_col = reward_col,
        relevant_dimension_col = relevant_dimension_col,
        initial_value = initial_value
      )

      sa_fit <- fit_feature_rl_model(
        data = train_data,
        option_cols = option_cols,
        model = "SA",
        start_params = c(start_alpha, start_phi, start_beta),
        block_col = block_col,
        choice_col = choice_col,
        reward_col = reward_col,
        relevant_dimension_col = relevant_dimension_col,
        initial_value = initial_value
      )

      ua_likelihood <- choice_likelihood(
        params = c(ua_fit$learning_rate, ua_fit$beta),
        data = test_data,
        option_cols = option_cols,
        model = "UA",
        block_col = block_col,
        choice_col = choice_col,
        reward_col = reward_col,
        relevant_dimension_col = relevant_dimension_col,
        initial_value = initial_value
      )

      sa_likelihood <- choice_likelihood(
        params = c(sa_fit$learning_rate, sa_fit$phi_unconstrained, sa_fit$beta),
        data = test_data,
        option_cols = option_cols,
        model = "SA",
        block_col = block_col,
        choice_col = choice_col,
        reward_col = reward_col,
        relevant_dimension_col = relevant_dimension_col,
        initial_value = initial_value
      )

      ua_results[[row_counter]] <- data.frame(
        model = "UA",
        heldout_block = heldout_block,
        iteration = iter,
        likelihood = ua_likelihood,
        learning_rate = ua_fit$learning_rate,
        phi = NA_real_,
        beta = ua_fit$beta,
        bic = ua_fit$bic,
        neg_log_likelihood = ua_fit$neg_log_likelihood
      )

      sa_results[[row_counter]] <- data.frame(
        model = "SA",
        heldout_block = heldout_block,
        iteration = iter,
        likelihood = sa_likelihood,
        learning_rate = sa_fit$learning_rate,
        phi = sa_fit$phi,
        beta = sa_fit$beta,
        bic = sa_fit$bic,
        neg_log_likelihood = sa_fit$neg_log_likelihood
      )

      row_counter <- row_counter + 1
    }
  }

  list(
    ua = do.call(rbind, ua_results),
    sa = do.call(rbind, sa_results),
    starting_params = do.call(rbind, starting_rows)
  )
}


summarize_cv_fits <- function(cv_results) {
  #' Summarize cross-validated model fits.
  #'
  #' @param cv_results Output from fit_models_leave_one_block_out().
  #'
  #' @return Dataframe with mean held-out likelihood and fitted parameters.
  ua <- cv_results$ua
  sa <- cv_results$sa

  rbind(
    data.frame(
      model = "UA",
      mean_likelihood = mean(ua$likelihood, na.rm = TRUE),
      mean_learning_rate = mean(ua$learning_rate, na.rm = TRUE),
      mean_phi = NA_real_,
      mean_beta = mean(ua$beta, na.rm = TRUE)
    ),
    data.frame(
      model = "SA",
      mean_likelihood = mean(sa$likelihood, na.rm = TRUE),
      mean_learning_rate = mean(sa$learning_rate, na.rm = TRUE),
      mean_phi = mean(sa$phi, na.rm = TRUE),
      mean_beta = mean(sa$beta, na.rm = TRUE)
    )
  )
}

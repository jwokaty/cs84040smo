#' Sequential Minimal Optimization model
#'
#' @description R6Class to maintain references to SMO elements.
#' Includes functions to initialize, print, and finalize (delete).
#'
#' @field x matrix or data.frame of observations
#' @field y data.frame, matrix, or vector of target values
#' @field kernel name
#' @field b bias
#' @field beta for sigmoid or gaussian kernel
#' @field theta for sigmoid kernel
#' @field d for polynomial kernel
#' @field epsilon sensitivity of margin to errors
#' @field tol convergence tolerance parameter
#' @field C margin
#' @field alpha vector
#' @field E error cache
#' @field error_cache_updates
#' @field verbose TRUE to print messages
#'
#' @examples
#' 
#' toy_x <- iris |>
#'   dplyr::select(-c(Species))
#' toy_y <- iris |>
#'   dplyr::mutate(y = dplyr::if_else(Species == "setosa", 1, -1)) |>
#'   dplyr::pull(y)
#' toy_model <- SmoModel$new(x = toy_x, y = toy_y, kernel = "linear")
#'
#' @export
SmoModel <- R6::R6Class("SmoModel", list(
  x = NULL,
  y = NULL,
  kernel = "linear",
  b = 0.0,
  beta = 1.0,
  d = 1,
  sigma = 1.0,
  theta = 1.0,
  epsilon = 10^(-3),
  tol = 10^(-3),  # convergence tolerance parameter
  C = 1.0,
  alpha = NULL,
  E = NULL,
  error_cache_updates = 0,
  verbose = FALSE,
  #' @description Initialize an SmoModel
  #' @param x matrix or data.frame of observations
  #' @param y data.frame, matrix, or vector of target values
  #' @param kernel name
  #' @param b bias
  #' @param beta for sigmoid or gaussian kernel
  #' @param sigma for gaussian kernel
  #' @param theta for sigmoid kernel
  #' @param d for polynomial kernel
  #' @param epsilon sensitivity of margin to errors
  #' @param tol convergence tolerance parameter
  #' @param C margin
  #' @param alpha vector
  #' @param E error cache
  #' @param error_cache_updates
  #' @param verbose TRUE to print messages
  initialize = function(x, y, kernel = "linear", b = 0.0, beta = 1, d = 1,
                        sigma = 1, theta = 1, epsilon = 10^(-3),
                        tol = 10^(-3), C = 1.0, verbose = FALSE) {
    
    if (kernel == "sigmoid")
      stopifnot(beta > 0 & theta > 0)
    else if (kernel == "gaussian")
      stopifnot(sigma > 0)
    else if (kernel == "polynomial")
      stopifnot(d >= 1)
    
    self$x <- as.matrix(x)
    self$y <- as.vector(y)
    self$kernel <- kernel
    self$b <- 0.0
    self$alpha <- rep(0.0, nrow(self$x))
    self$beta <- beta
    self$d <- d
    self$sigma <- sigma
    self$theta <- theta
    self$epsilon <- epsilon
    self$tol <- tol
    self$C <- C
    self$E <- rep(0.0, nrow(self$x))  # Initialize to -y instead of NA
    for (i in seq_len(nrow(self$x))) {
      self$E[i] <- -self$y[i]  # f(x) = 0 initially, so E = 0 - y = -y
    }
    self$error_cache_updates <- 0
    self$verbose <- verbose
  },
  #' @description print information about the instance
  print = function(...) {
    cat("SmoModel:\n")
    cat("  Kernel:", self$kernel, "\n")
    cat("  n samples:", length(self$y), "\n")
    cat("  n support vectors:", sum(self$alpha > 1e-5), "\n")
    cat("  b:", self$b, "\n")
    cat("  Alpha range: [", min(self$alpha), ",", max(self$alpha), "]\n")
    invisible(self)
  },
  #' @description call to delete instance
  finalize = function() {
    message("Deleting model")
  }
))

#' @description Kernel function
#'
#' @param x matrix or data.frame
#' @param y matrix or data.frame
#'
#' @return scalar
#'
#' @examples
#' 
#' toy_x <- iris |>
#'   dplyr::select(-c(Species))
#' toy_y <- iris |>
#'   dplyr::mutate(y = dplyr::if_else(Species == "setosa", 1, -1)) |>
#'   dplyr::pull(y)
#' toy_model <- SmoModel$new(x = toy_x, y = toy_y, kernel = "linear")
#' toy_model$K(toy_model$x[1, ], toy_model$x[2, ])
#'
#' @export
SmoModel$set("public", "K", function(x, y = NULL) {
  if (is.null(y))
    y <- x
  
  x <- as.vector(x)
  y <- as.vector(y)
  
  if (self$kernel == "polynomial") {
    z <- (sum(x * y) + 1)^self$d
  } else if (self$kernel == "gaussian") {
    z <- exp(-sum((x - y)^2) / (2 * self$sigma^2))
  } else if (self$kernel == "sigmoid") {
    z <- tanh(self$beta * sum(x * y) + self$theta)
  } else {  # linear
    z <- sum(x * y)
  }
  z
})

#' @description Dual function
#'
#' Equation: w(a) = sum(a) - .5 * sum(sum(yi * yj * ai * aj * K(xi, xj)))
#'
#' @param alpha vector
#'
#' @return scalar
#'
#' @examples
#' 
#' toy_x <- iris |>
#'   dplyr::select(-c(Species))
#' toy_y <- iris |>
#'   dplyr::mutate(y = dplyr::if_else(Species == "setosa", 1, -1)) |>
#'   dplyr::pull(y)
#' toy_model <- SmoModel$new(x = toy_x, y = toy_y, kernel = "linear")
#' toy_model$dual(toy_model$alpha)
#'
#' @export
SmoModel$set("public", "dual", function(alpha) {
  indices <- seq(1, length(alpha))
  nonzero_indices <- which(alpha > 1e-10)
  as_ys_Ks <- 0.0
  for (i in nonzero_indices) {
    for (j in nonzero_indices) {
      ai_aj <- alpha[i] * alpha[j]
      yi_yj <- self$y[i] * self$y[j]
      K_xi_xj <- self$K(self$x[i, ], self$x[j, ])
      as_ys_Ks <- as_ys_Ks + ai_aj * yi_yj * K_xi_xj
    }
  }
  w <- sum(alpha) - 0.5 * as_ys_Ks 
  w
})

#' @description Calculate f
#'
#' Equation: f(xi) = sum(aj * yj * K_xi_xj) + b
#'
#' @param i index of x relevant to E to be calculated
#' @param v matrix or data.frame
#'
#' @return scalar
#'
#' @examples
#' 
#' toy_x <- iris |>
#'   dplyr::select(-c(Species))
#' toy_y <- iris |>
#'   dplyr::mutate(y = dplyr::if_else(Species == "setosa", 1, -1)) |>
#'   dplyr::pull(y)
#' toy_model <- SmoModel$new(x = toy_x, y = toy_y, kernel = "linear")
#' toy_model$f(1)
#' 
#' @export
SmoModel$set("public", "f", function(i, v = NULL) {
  if (is.null(v))
    v <- self$x
  
  v <- as.matrix(v)
  K_xi_vj <- apply(v, 1, function(vj) {
    self$K(self$x[i, ], vj)
  })
  ayKs <- sum(self$alpha * self$y * K_xi_vj)
  fx <- ayKs + self$b
  fx
})

#' @description Calculate E
#'
#' Equation: Ei = f(xi) - yi = sum(aj * yj * K_xi_xj + b) - yi
#'
#' @param i index of x relevant to E to be calculated
#'
#' @return scalar
#'
#' @examples
#' 
#' toy_x <- iris |>
#'   dplyr::select(-c(Species))
#' toy_y <- iris |>
#'   dplyr::mutate(y = dplyr::if_else(Species == "setosa", 1, -1)) |>
#'   dplyr::pull(y)
#' toy_model <- SmoModel$new(x = toy_x, y = toy_y, kernel = "linear")
#' toy_model$calculateE(2)
#'
#' @export
SmoModel$set("public", "calculateE", function(i) {
  Ei <- self$f(i) - self$y[i]
  Ei
})

#' SMO update function
#'
#' @param i1 corresponding index for alpha
#' @param i2 corresponding index for alpha
#'
#' @return scalar
#'
#' @examples
#' 
#' toy_x <- iris |>
#'   dplyr::select(-c(Species))
#' toy_y <- iris |>
#'   dplyr::mutate(y = dplyr::if_else(Species == "setosa", 1, -1)) |>
#'   dplyr::pull(y)
#' toy_model <- SmoModel$new(x = toy_x, y = toy_y, kernel = "linear")
#' toy_model$takeStep(1, 2)
#'
#' @export
SmoModel$set("public", "takeStep", function(i1, i2) {
  if (i1 == i2)
    return(0)
  
  alpha1 <- self$alpha[i1]
  alpha2 <- self$alpha[i2]
  E1 <- self$E[i1]
  E2 <- self$E[i2]
  y1 <- self$y[i1]
  y2 <- self$y[i2]
  s <- y1 * y2
  
  # Compute L and H
  if (y1 == y2) {
    L <- max(0, alpha2 + alpha1 - self$C)
    H <- min(self$C, alpha2 + alpha1)
  } else {
    L <- max(0, alpha2 - alpha1)
    H <- min(self$C, self$C + alpha2 - alpha1) 
  }
  
  if (L == H)
    return(0)
  
  k11 <- self$K(self$x[i1, ], self$x[i1, ])
  k12 <- self$K(self$x[i1, ], self$x[i2, ])
  k22 <- self$K(self$x[i2, ], self$x[i2, ])
  eta <- k11 + k22 - 2 * k12
  
  if (eta > 0) {
    new_alpha2 <- alpha2 + (y2 * (E1 - E2)) / eta
    # Clip
    if (new_alpha2 < L) 
      new_alpha2 <- L
    else if (new_alpha2 > H)
      new_alpha2 <- H
  } else {
    # Evaluate objective function at endpoints
    LH_alpha2 <- self$alpha
    LH_alpha2[i2] <- L
    Lobj <- self$dual(LH_alpha2)
    LH_alpha2[i2] <- H
    Hobj <- self$dual(LH_alpha2)
    
    if (Lobj < (Hobj - self$epsilon))
      new_alpha2 <- L
    else if (Lobj > (Hobj + self$epsilon))
      new_alpha2 <- H
    else
      new_alpha2 <- alpha2
  }
  
  if (abs(new_alpha2 - alpha2) < self$epsilon * 
      (new_alpha2 + alpha2 + self$epsilon))
    return(0)
  
  # Find new alpha1
  new_alpha1 <- alpha1 + s * (alpha2 - new_alpha2)
  
  if (new_alpha1 < 0) {
     new_alpha2 <- new_alpha2 + s * new_alpha1
     new_alpha1 <- 0
     # Clip new_alpha2 if out of bounds
     new_alpha2 <- max(0, min(self$C, new_alpha2))
  } else if (new_alpha1 > self$C) {
     new_alpha2 <- new_alpha2 + s * (new_alpha1 - self$C)
     new_alpha1 <- self$C
     # Clip new_alpha2 if out of bounds
     new_alpha2 <- max(0, min(self$C, new_alpha2))
  }

  # May be redundant
  new_alpha1 <- max(0, min(self$C, new_alpha1))
  new_alpha2 <- max(0, min(self$C, new_alpha2))

  # Skip very small changes
  if (abs(new_alpha1 - alpha1) < 1e-10 & abs(new_alpha2 - alpha2) < 1e-10)
     return(0)

  # Update bias
  b1 <- self$b - E1 - y1 * (new_alpha1 - alpha1) * k11 - y2 * (new_alpha2 - alpha2) * k12
  b2 <- self$b - E2 - y1 * (new_alpha1 - alpha1) * k12 - y2 * (new_alpha2 - alpha2) * k22
  old_b <- self$b
  if (0 < new_alpha1 & new_alpha1 < self$C)
    self$b <- b1
  else if (0 < new_alpha2 & new_alpha2 < self$C)
    self$b <- b2
  else
    self$b <- (b1 + b2) / 2

  # Update alphas first
  self$alpha[i1] <- new_alpha1
  self$alpha[i2] <- new_alpha2
  
  # Update error cache for all points
  delta_b <- self$b - old_b
  for (i in seq_along(self$E)) {
    self$E[i] <- self$E[i] + 
      y1 * (new_alpha1 - alpha1) * self$K(self$x[i1, ], self$x[i, ]) +
      y2 * (new_alpha2 - alpha2) * self$K(self$x[i2, ], self$x[i, ]) +
      delta_b
  }

  self$error_cache_updates <- self$error_cache_updates + 1
  if (self$error_cache_updates %% 500 == 0) {
    for (i in seq_along(self$E))
      self$E[i] <- self$calculateE(i)
    if (self$verbose)
      print("Refreshed error cache")
  }
  
  return(1)
})

#' @export
SmoModel$set("public", "boundAlphaIndex", function(bound, index_type) {
  index <- NA
  best_E <- NA
  for (i in bound) {
    if (is.na(best_E) |
        (index_type == "max" & self$E[i] > best_E) |
        (index_type == "min" & self$E[i] < best_E)) {
      best_E <- self$E[i]
      index <- i
    }
  }
  return(index)
})

#' SMO inner loop
#'
#' Select best i1 in relation to i2
#'
#' @param i2 corresponding index for alpha
#'
#' @return scalar
#'
#' @examples
#' 
#' toy_x <- iris |>
#'   dplyr::select(-c(Species))
#' toy_y <- iris |>
#'   dplyr::mutate(y = dplyr::if_else(Species == "setosa", 1, -1)) |>
#'   dplyr::pull(y)
#' toy_model <- SmoModel$new(x = toy_x, y = toy_y, kernel = "linear")
#' toy_model$examineExamples(2)
#'
#' @export
SmoModel$set("public", "examineExamples", function(i2) {
  y2 <- self$y[i2]
  alpha2 <- self$alpha[i2]
  E2 <- self$E[i2]
  r2 <- E2 * y2
  
  # Check for KKT violation
  if ((r2 < -self$tol & alpha2 < self$C) | (r2 > self$tol & alpha2 > 0)) {
    # First, try non-bound examples with maximum step size heuristic
    bound <- which(self$alpha > 0 & self$alpha < self$C)
    if (length(bound) > 1) {
      # Choose multiplier with maximum step size
      if (E2 > 0) {
        i1 <- self$boundAlphaIndex(bound, "min")
      } else {
        i1 <- self$boundAlphaIndex(bound, "max")
      }
      if (!is.na(i1) && self$takeStep(i1, i2))
        return(1)
      
      # Loop through all bound examples starting at random point
      start_idx <- sample(length(bound), 1)
      for (j in seq_along(bound)) {
        i1 <- bound[((start_idx + j - 2) %% length(bound)) + 1]
        if (self$takeStep(i1, i2))
          return(1)
      }
    }
    
    # Loop through all examples starting at random point
    n <- length(self$alpha)
    start_idx <- sample(n, 1)
    for (j in seq_len(n)) {
      i1 <- ((start_idx + j - 2) %% n) + 1
      if (self$takeStep(i1, i2))
        return(1)
    }
  }
  0
})

#' SMO outerloop function to select i2
#'
#' @param model SmoModel
#' @param max_iterations (default 10000)
#'
#' @return SmoModel
#'
#' @examples
#' 
#' toy_x <- iris |>
#'   dplyr::select(-c(Species))
#' toy_y <- iris |>
#'   dplyr::mutate(y = dplyr::if_else(Species == "setosa", 1, -1)) |>
#'   dplyr::pull(y)
#' toy_model <- SmoModel$new(x = toy_x, y = toy_y, kernel = "linear")
#' fit_smo(toy_model)
#'
#' @export
fit_smo <- function(model, max_iterations = 10000) {
  numChanged <- 0
  examineAll <- 1
  iteration <- 0
  start_time <- Sys.time()
  
  while ((numChanged > 0 | examineAll == 1) & iteration < max_iterations) {
    numChanged <- 0
    
    if (examineAll == 1) {
      # Loop over all training examples
      for (i in seq_along(model$alpha)) {
        numChanged <- numChanged + model$examineExamples(i)
      }
    } else {
      # Loop over examples where alpha is not 0 or C (non-bound examples)
      non_bound <- which(model$alpha > 0 & model$alpha < model$C)
      for (i in non_bound) {
        numChanged <- numChanged + model$examineExamples(i)
      }
    }
    
    if (examineAll == 1) {
      examineAll <- 0
    } else if (numChanged == 0) {
      examineAll <- 1
    }
    
    if (model$verbose & iteration %% 50 == 0) {
      elapsed <- as.numeric(Sys.time() - start_time)
      cat(sprintf("Iteration: %4d | Changed: %3d | SVs: %4d | Time: %ds\n",
           iteration, numChanged, sum(model$alpha > 1e-5), round(elapsed)))
    }
    iteration <- iteration + 1
  }
  
  if (iteration >= max_iterations) {
      warning("Reached maximum iterations without convergence.")
  }
  
  if (!model$verbose) {
    cat("Training complete. Support vectors:", sum(model$alpha > 1e-5), "/", 
        length(model$alpha), "\n")
  }
  
  model
}

#' Prediction function
#'
#' Equation: y = sum(self$alpha * self$y * K_v_i + self$b)
#'
#' @param v matrix or data.frame of observations to predict
#'
#' @return a vector of 1 and -1
#'
#' @examples
#' 
#' toy_x <- iris |>
#'   dplyr::select(-c(Species))
#' toy_y <- iris |>
#'   dplyr::mutate(y = dplyr::if_else(Species == "setosa", 1, -1)) |>
#'   dplyr::pull(y)
#' toy_model <- SmoModel$new(x = toy_x, y = toy_y, kernel = "linear")
#' train_smo(toy_model)
#' predictions <- toy_model$predict(toy_x)
#'
#' @export
SmoModel$set("public", "predict", function(v) {
  v <- as.matrix(v)
  y <- numeric(nrow(v))
  
  for (i in seq_len(nrow(v))) {
    # Compute prediction for new point v[i,]
    K_v <- apply(self$x, 1, function(xj) {
      self$K(v[i, ], xj)
    })
    raw_prediction <- sum(self$alpha * self$y * K_v) + self$b
    y[i] <- ifelse(raw_prediction >= 0, 1, -1)
  }
  y
})

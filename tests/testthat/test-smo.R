toy_x <- iris |>
  dplyr::filter(Species != "versicolor") |>
  dplyr::select(-c(Species))
toy_y <- iris |>
  dplyr::filter(Species != "versicolor") |>
  dplyr::mutate(y = dplyr::if_else(Species == "virginica", 1, -1)) |>
  dplyr::pull(y)
toy_model <- SmoModel$new(x = toy_x, y = toy_y, kernel = "linear",
                          verbose = FALSE)
toy_model$alpha[1] <- 1

sm_x <- data.frame(x1 = c(1, 2, 0, 1),
                   x2 = c(1, 2, 0, 0))
sm_y <- c(1, 1, -1, -1)
alpha <- c(.5, 1, 0, 2)
small_model <- SmoModel$new(sm_x, sm_y, kernel = "linear",
                            verbose = FALSE, C = 1.0)

test_that("K(xi, xj) returns a scalar", {
  x <- c(1, 2)
  y <- c(3, 4)
  expect_equal(toy_model$K(x), 5)
  expect_equal(toy_model$K(x, y), 11)
  expect_equal(toy_model$K(toy_model$x[1, ],toy_model$x[2, ]), 37.49)
})

test_that("f calculates f(x[1, ])", {
  expect_equal(toy_model$f(1), -40.26)

  small_model$alpha <- c(.5, 1, 0, 2)
  expect_equal(small_model$f(1), 3)
  expect_equal(small_model$f(4), 0.5)
  small_model$alpha <- 0
})

test_that("calculateE gives f(x[1, ]) - y[1]", {
  expect_equal(toy_model$calculateE(1), -39.26)
})

test_that("dual gives f(x[1, ]) - y[1]", {
  expect_equal(small_model$dual(alpha), 0.25)
})

test_that("boundAlphaIndex gives max or min bound alpha index", {
  small_model2 <- SmoModel$new(sm_x, sm_y, kernel = "linear",
                               verbose = FALSE, C = 2.0)
  small_model2$alpha <- c(.5, 1, .2, 2)
  bound <- which(0 < small_model2$alpha &
                 small_model2$alpha < small_model2$C)
  expect_equal(small_model2$boundAlphaIndex(bound, "min"), 1)
  expect_equal(small_model2$boundAlphaIndex(bound, "max"), 3)
})

test_that("predict yields 1 or -1", {
  sm_x <- data.frame(x1 = c(1, 2, 0, 1),
                     x2 = c(1, 2, 0, 0))
  sm_y <- c(1, 1, -1, -1)
  small_model3 <- SmoModel$new(sm_x, sm_y, kernel = "linear",
                               verbose = FALSE, C = 1.0)
  fit_smo(small_model3)
  new_x1_x2 <- data.frame(x1 = 0, x2 = 0)
  expect_equal(small_model3$predict(new_x1_x2), -1)
})

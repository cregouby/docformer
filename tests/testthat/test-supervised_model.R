test_that("docformer config works with default params", {
  expect_error(config  <-  docformer_config(), NA)
})

test_that("docformer config works with pretrained model params", {
  pretrained_model_name <- "hf-internal-testing/tiny-layoutlm"
  expect_error(
    config  <-  docformer_config(pretrained_model_name=pretrained_model_name),
    NA)
  expect_equal(config$hidden_size, 32L)
  expect_equal(config$vocab_size, 5000L)
})


test_that("docformer config warn if wrong model name", {
  pretrained_model_name <- "inst/tiny-layoutlm"
  expect_warning(
    config  <-  docformer_config(pretrained_model_name=pretrained_model_name),
    "Provided model name cannot be found")
  expect_equal(config$hidden_size, 768L)
  expect_equal(config$vocab_size, 30522L)
})



test_that("docformer initialization works with default params", {
  config  <-  docformer_config()
  expect_error(docformer_net <- docformer(config), NA)
})


test_that("docformer works ", {
  config = docformer_config()
  expect_error(docformer_net <- docformer(config), NA)
})


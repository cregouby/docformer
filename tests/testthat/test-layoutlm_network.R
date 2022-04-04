test_that("LayoutLMForTokenClassification initialize works ", {
  config  <-  docformer_config()
  expect_error(layoutlm_net <- LayoutLMForTokenClassification(config), NA)
  expect_s3_class(layoutlm_net, "LayoutLMPreTrainedModel")
  expect_s3_class(layoutlm_net, "nn_module")
  expect_true(torch::is_nn_module(layoutlm_net))
  expect_true(torch::is_nn_module(layoutlm_net$layoutlm))
  expect_true(torch::is_nn_module(layoutlm_net$layoutlm$embeddings))
  expect_true(torch::is_nn_module(layoutlm_net$layoutlm$encoder))
  expect_true(torch::is_nn_module(layoutlm_net$layoutlm$pooler))
})

test_that("LayoutLMForTokenClassification from_pretrain works from local file", {
  skip_on_cran()
  skip_on_os("windows")
  config  <-  docformer_config()
  pretrained_model_name <- here::here("inst/layoutlm-base-uncased.pth")
  layoutlm_net <- LayoutLMForTokenClassification(config)
  expect_error(layoutlm_mod <- layoutlm_net$from_pretrained(pretrained_model_name=pretrained_model_name), NA)
})

test_that("LayoutLMForTokenClassification from_pretrain works from public weights", {
  skip_on_cran()
  skip_on_os("windows")
  config  <-  docformer_config()
  pretrained_model_name <- "hf-internal-testing/tiny-layoutlm"
  layoutlm_net <- LayoutLMForTokenClassification(config)
  expect_error(layoutlm_mod <- layoutlm_net$from_pretrained(pretrained_model_name=pretrained_model_name), NA)
})

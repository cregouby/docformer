test_that("LayoutLMForTokenClassification initialize works ", {
  config  <-  docformer_config()
  expect_error(layoutlm_net <- LayoutLMForTokenClassification(config), NA)
})

test_that("LayoutLMForTokenClassification from_pretrain works ", {
  pretrained_model_name <- "microsoft/layoutlm-base-uncased"
  expect_error(layoutlm_mod <- LayoutLMForTokenClassification$from_pretrain(pretrained_model_name=pretrained_model_name), NA)
})

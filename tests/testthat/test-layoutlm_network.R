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

test_that("LayoutLMForTokenClassification from_pretrain works ", {
  pretrained_model_name <- "microsoft/layoutlm-base-uncased"
  expect_error(layoutlm_mod <- LayoutLMForTokenClassification$from_pretrain(pretrained_model_name=pretrained_model_name), NA)
})

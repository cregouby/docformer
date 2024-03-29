test_that("docformer_for_masked_lm unitary functions works", {
  config  <-  docformer_config(pretrained_model_name = "allenai/hvila-row-layoutlm-finetuned-docbank")
  docformer_net <- docformer:::docformer_for_masked_lm(config, .mask_id(bpe_tok_mask))
  embedding <- docformer_net$docformer(mask_for_mm_mlm(doc_tt, .mask_id(bpe_tok_mask)))
  # expect_tensor_shape(embedding, c(2, config$max_position_embeddings, config$hidden_size))
  # mm_mlm
  expect_no_error(mm_mlm <- docformer_net$mm_mlm(embedding))
  expect_tensor_shape(mm_mlm, c(2, config$max_position_embeddings, config$vocab_size))
  expect_tensor_dtype(mm_mlm, "Float")
  # ltr
  expect_no_error(ltr <- docformer_net$ltr(embedding))
  expect_no_error(ltr_loss <- docformer_net$ltr_loss(torch::nnf_interpolate(ltr, doc_tt$image$shape[3:4]), doc_tt$image))
  # tdi
  expect_no_error(tdi <- docformer_net$tdi(embedding))
  expect_tensor_shape(tdi, c(2, 1))
  expect_gte(as.numeric(tdi$min()), 0)
  expect_lte(as.numeric(tdi$max()), 1)

})

test_that("docformer_for_masked_lm on non-default design raise an exception", {
  config  <-  docformer_config(pretrained_model_name = "hf-internal-testing/tiny-layoutlm")
  expect_error(docformer_net <- docformer:::docformer_for_masked_lm(config, .mask_id(bpe_tok_mask)),
               "hidden_size"
  )
})

test_that("docformer_for_masked_lm works", {
  skip_on_os(os = c("windows", "mac", "solaris"))
  config  <-  docformer_config(pretrained_model_name = "allenai/hvila-row-layoutlm-finetuned-docbank")
  # tic()
  docformer_net <- docformer:::docformer_for_masked_lm(config, .mask_id(bpe_tok_mask))
  # toc() # 29s
  expect_no_error(
    # tic()
    result <- docformer_net(doc_tt)
    # toc() # 234s
  )
  expect_tensor_dtype(result$loss, "Float")

})

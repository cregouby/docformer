
test_that("docformer_config(dtype = torch::torch_float16()) works", {
  config16  <-  docformer_config( dtype = torch::torch_float16() )
  # config32  <-  docformer_config( dtype = torch::torch_float32() )
  # config64  <-  docformer_config( dtype = torch::torch_float64() )
  expect_no_error(docformer_16 <- docformer:::docformer(config16))
  # model is too big, goes to swap
  # expect_no_error(docformer_32 <- docformer:::docformer(config32))
  expect_no_error(output_16 <- docformer_16(doc_tt))
  expect_tensor_dtype(output_16, torch::torch_float16())
  # expect_no_error(output_32 <- docformer_32(doc_tt))
  # expect_no_error(docformer_64 <- docformer:::docformer(config64)) # OOM & Swap
  # expect_equal(lobstr::obj_size(output_32), lobstr::obj_size(output_16) * 2)
  # expect_gt(lobstr::obj_size(docformer_64), lobstr::obj_size(docformer_32))
})

test_that("docformer_for_masked_lm unitary functions work", {
  # config  <-  docformer_config(pretrained_model_name = "hf-internal-testing/tiny-layoutlm")
  config  <-  docformer_config(pretrained_model_name = "allenai/hvila-row-layoutlm-finetuned-docbank", dtype = torch::torch_float16())
  docformer_net <- docformer:::docformer_for_masked_lm(config)
  embedding <- docformer_net$docformer(mask_for_mm_mlm(x))
  # expect_tensor_shape(embedding, c(2, config$max_position_embeddings, config$hidden_size))
  # mm_mlm
  expect_no_error(mm_mlm <- docformer_net$mm_mlm(embedding))
  expect_tensor_shape(mm_mlm, c(2, config$max_position_embeddings, config$vocab_size))
  expect_tensor_dtype(mm_mlm, torch::torch_float16())
  # ltr
  expect_no_error(ltr <- docformer_net$ltr(embedding[,1,]))
  expect_tensor_shape(ltr, c(2, config$max_position_embeddings, config$hidden_size))
  # tdi
  expect_no_error(tdi <- docformer_net$tdi(embedding))
  expect_tensor_shape(tdi, c(2, config$max_position_embeddings, config$hidden_size))


})

test_that("docformer initialization works with default params", {
  config  <-  docformer_config()
  expect_no_error(docformer_net <- docformer:::docformer(config))
  #  validation on the output of extract_features layers
  expect_equal(length(docformer_net$encoder$layers$children), config$num_hidden_layers)
  expect_tensor_shape(docformer_net$extract_feature$visual_feature(doc_tt$image), c(2, config$max_position_embeddings, config$hidden_size))
  expect_tensor_shape(docformer_net$extract_feature$language_feature(doc_tt$text), c(2, config$max_position_embeddings, config$hidden_size))
  expect_error(spatial_tt <- docformer_net$extract_feature$spatial_feature(doc_tt$x_features, doc_tt$y_features), NA)
  expect_tensor_shape(spatial_tt[[1]], c(2, config$max_position_embeddings, config$hidden_size))
  expect_tensor_shape(spatial_tt[[2]], c(2, config$max_position_embeddings, config$hidden_size))
})

test_that("docformer initialization works with standard pretrained model by name", {
  config  <-  docformer_config(pretrained_model_name = "allenai/hvila-row-layoutlm-finetuned-docbank")
  expect_no_error(docformer_net <- docformer:::docformer(config))
  #  validation on the output of extract_features layers
  expect_equal(length(docformer_net$encoder$layers$children), config$num_hidden_layers)
  expect_tensor_shape(docformer_net$extract_feature$visual_feature(doc_tt$image), c(2, config$max_position_embeddings, config$hidden_size))
  expect_tensor_shape(docformer_net$extract_feature$language_feature(doc_tt$text), c(2, config$max_position_embeddings, config$hidden_size))
  expect_error(spatial_tt <- docformer_net$extract_feature$spatial_feature(doc_tt$x_features, doc_tt$y_features), NA)
  expect_tensor_shape(spatial_tt[[1]], c(2, config$max_position_embeddings, config$hidden_size))
  expect_tensor_shape(spatial_tt[[2]], c(2, config$max_position_embeddings, config$hidden_size))
})

test_that("docformer initialization works with non-standard pretrained model by name", {
  config  <-  docformer_config(pretrained_model_name = "hf-internal-testing/tiny-layoutlm")
  expect_no_error(docformer_net <- docformer:::docformer(config))
  #  validation on the output of extract_features layers
  expect_equal(length(docformer_net$encoder$layers$children), config$num_hidden_layers)
  # TODO cannot match image embedding shape and config$hidden_size %/% config$intermediate_ff_size_factor with this network
  # Error :  mat1 and mat2 shapes cannot be multiplied (64x30 and 16x512)
  # expect_tensor_shape(docformer_net$extract_feature$visual_feature(tiny_tt$image), c(2, config$max_position_embeddings, config$hidden_size))
  expect_tensor_shape(docformer_net$extract_feature$language_feature(tiny_tt$text), c(2, config$max_position_embeddings, config$hidden_size))
  expect_no_error(spatial_tt <- docformer_net$extract_feature$spatial_feature(tiny_tt$x_features, tiny_tt$y_features))
  expect_tensor_shape(spatial_tt[[1]], c(2, config$max_position_embeddings, config$hidden_size))
  expect_tensor_shape(spatial_tt[[2]], c(2, config$max_position_embeddings, config$hidden_size))
})

# test_that("docformer initialization works with manual parameter values", {
#   config_man  <-
#     docformer_config(
#       coordinate_size = 8L,
#       shape_size = 8L,
#       hidden_size = 48L,
#       max_2d_position_embeddings = 128L,
#       max_position_embeddings = 64L,
#       num_attention_heads = 2L,
#       num_hidden_layers = 2L,
#       vocab_size = 5000L,
#       intermediate_ff_size_factor = 2L
#     )
#   # TODO Error The size of tensor a (48) must match the size of tensor b (768) at non-singleton dimension 1
#   expect_no_error(docformer_net <- docformer:::docformer(config_man))
#   expect_no_error(output_tt <- docformer_net(tiny_tt))
# })

test_that("docformer forward works with the expected tensor input", {
  # config  <-  docformer_config(pretrained_model_name = "hf-internal-testing/tiny-layoutlm")
  config  <-  docformer_config(pretrained_model_name = "allenai/hvila-row-layoutlm-finetuned-docbank")
  docformer_net <- docformer:::docformer(config)
  expect_no_error(output_tt <- docformer_net(doc_tt))
  expect_tensor_shape(output_tt, c(2, config$max_position_embeddings, config$hidden_size))
})

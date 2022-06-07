test_that("docformer initialization works with default params", {
  config  <-  docformer_config()
  expect_error(docformer_net <- docformer(config), NA)
  #  validation on the output of extract_features layers
  expect_equal(length(docformer_net$encoder$layers$children), config$num_hidden_layers)
  expect_equal(docformer_net$extract_feature$visual_feature(doc_tt$image)$shape, c(2, config$max_position_embeddings, config$hidden_size))
  expect_equal(docformer_net$extract_feature$language_feature(doc_tt$text)$shape, c(2, config$max_position_embeddings, config$hidden_size))
  expect_error(spatial_tt <- docformer_net$extract_feature$spatial_feature(doc_tt$x_features, doc_tt$y_features), NA)
  expect_equal(spatial_tt[[1]]$shape, c(2, config$max_position_embeddings, config$hidden_size))
  expect_equal(spatial_tt[[2]]$shape, c(2, config$max_position_embeddings, config$hidden_size))
})

test_that("docformer initialization works with standard pretrained model by name", {
  config  <-  docformer_config(pretrained_model_name = "allenai/hvila-row-layoutlm-finetuned-docbank")
  expect_error(docformer_net <- docformer(config), NA)
  #  validation on the output of extract_features layers
  expect_equal(length(docformer_net$encoder$layers$children), config$num_hidden_layers)
  expect_equal(docformer_net$extract_feature$visual_feature(doc_tt$image)$shape, c(2, config$max_position_embeddings, config$hidden_size))
  expect_equal(docformer_net$extract_feature$language_feature(doc_tt$text)$shape, c(2, config$max_position_embeddings, config$hidden_size))
  expect_error(spatial_tt <- docformer_net$extract_feature$spatial_feature(doc_tt$x_features, doc_tt$y_features), NA)
  expect_equal(spatial_tt[[1]]$shape, c(2, config$max_position_embeddings, config$hidden_size))
  expect_equal(spatial_tt[[2]]$shape, c(2, config$max_position_embeddings, config$hidden_size))
})

test_that("docformer initialization works with non-standard pretrained model by name", {
  config  <-  docformer_config(pretrained_model_name = "hf-internal-testing/tiny-layoutlm")
  expect_error(docformer_net <- docformer(config), NA)
  #  validation on the output of extract_features layers
  expect_equal(length(docformer_net$encoder$layers$children), config$num_hidden_layers)
  expect_equal(docformer_net$extract_feature$visual_feature(doc_tt$image)$shape, c(2, config$max_position_embeddings, config$hidden_size))
  expect_equal(docformer_net$extract_feature$language_feature(doc_tt$text$clamp_max(config$vocab_size))$shape, c(2, config$max_position_embeddings, config$hidden_size))
  expect_error(spatial_tt <- docformer_net$extract_feature$spatial_feature(doc_tt$x_features, doc_tt$y_features), NA)
  expect_equal(spatial_tt[[1]]$shape, c(2, config$max_position_embeddings, config$hidden_size))
  expect_equal(spatial_tt[[2]]$shape, c(2, config$max_position_embeddings, config$hidden_size))
})

test_that("docformer forward works with the expected tensor input", {
  config  <-  docformer_config()
  docformer_net <- docformer(config)
  expect_error(output_tt <- docformer_net(doc_tt), NA)
  expect_equal(output_tt$shape, c(2, config$max_position_embeddings, config$hidden_size))
})


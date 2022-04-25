test_that("docformer initialization works with default params", {
  config  <-  docformer_config()
  expect_error(docformer_net <- docformer(config), NA)
  # TODO additional validation on the output
  expect_equal(length(docformer_net$encoder$layers$children), config$num_hidden_layers)
})

test_that("docformer forward works with the expected tensor input", {
  config  <-  docformer_config()
  docformer_net <- docformer(config)
  doc_tt <- create_features_from_doc(doc, sent_tok_mask)
  expect_error(output_tt <- docformer_net(doc_tt), NA)
})


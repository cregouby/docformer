sent_tok <- sentencepiece::sentencepiece_load_model(system.file(package="sentencepiece", "models/nl-fr-dekamer.model"))
sent_tok_mask <- sent_tok
sent_tok_mask$vocab_size <- sent_tok_mask$vocab_size+1L
sent_tok_mask$vocabulary <- rbind(sent_tok_mask$vocabulary, data.frame(id=sent_tok_mask$vocab_size, subword="<mask>"))

bpe_tok <- tokenizers.bpe::bpe_load_model(system.file(package="tokenizers.bpe", "extdata/youtokentome.bpe"))
bpe_tok_mask <- bpe_tok
bpe_tok_mask$vocab_size <- bpe_tok_mask$vocab_size+1L
bpe_tok_mask$vocabulary <- rbind(bpe_tok_mask$vocabulary, data.frame(id=bpe_tok_mask$vocab_size, subword="<MASK>"))

# hf_tok <- hftokenizers::(system.file(package="sentencepiece", "models/nl-fr-dekamer.model"))

image <- system.file(package="docformer", "inst", "2106.11539_1.png")
doc <- system.file(package="docformer", "inst", "2106.11539_1_2.pdf")

test_that("normalize_box works with single var", {
  expect_equal(normalize_box(c(22,34,27,41), width=100, height=100, size=100), c(22,34,27,41))
})

# test_that("normalize_box works with dataframe", {
#   df <- tibble::tibble(bbox=c(list(22,34,27,41),list(22,34,27,41)),  width=c(100,100), height=c(100,100), size=c(100,100))
#   expect_equal(all(normalize_box(df), c(22,34,27,41)))
# })

test_that(".tokenize return a flat list for sentencepiece and tokenizers.bpe", {
  phrase <- c("This", "Simple", "Coconut", "Curry", "Recipe", "Produces", "Flavorful", "Fish", "Fast")
  # sentencepiece
  expect_true(purrr::vec_depth(.tokenize(tokenizer=sent_tok, phrase))==2)
  # hf tokenizer
  # expect_true(purrr::vec_depth(.tokenize(tokenizer=hf_tok, phrase))==2)
  # tokenizer.bpe
  expect_true(purrr::vec_depth(.tokenize(tokenizer=bpe_tok, phrase))==2)

})

test_that("tokenizer that cannot encode MASK raise an error", {
  # sentencepiece
  expect_error(.mask_id(sent_tok), regexp = "tokenizer do not encode `<")
  # hf tokenizer
  # expect_error(.mask_id(hf_tok), regexp = "tokenizer do not encode <")
  # tokenizer.bpe
  expect_error(.mask_id(bpe_tok), regexp = "tokenizer do not encode `<")

})

test_that("create_features_from_image works with default values", {
  # sentencepiece
  expect_error(create_features_from_image(image, sent_tok_mask),
               regexp = NA)
  # hf tokenizer
  # expect_error(.mask_id(hf_tok), regexp = "tokenizer do not encode <")
  # tokenizer.bpe
  expect_error(create_features_from_image(image, bpe_tok_mask),
               regexp = NA)
})

test_that("create_features_from_doc works with default values", {
  # sentencepiece
  expect_error(create_features_from_doc(doc, sent_tok_mask),
               regexp = NA)
  # hf tokenizer
  # expect_error(.mask_id(hf_tok), regexp = "tokenizer do not encode <")
  # tokenizer.bpe
  expect_error(create_features_from_doc(doc, bpe_tok_mask),
               regexp = NA)
})


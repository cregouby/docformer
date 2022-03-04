sent_tok <- sentencepiece::sentencepiece_load_model(system.file(package="sentencepiece", "models/nl-fr-dekamer.model"))
bpe_tok <- tokenizers.bpe::bpe_load_model(system.file(package="tokenizers.bpe", "extdata/youtokentome.bpe"))
# hf_tok <- hftokenizers::(system.file(package="sentencepiece", "models/nl-fr-dekamer.model"))

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

test_that(".tokenize cannot encode MASK raise an error", {
  # sentencepiece
  expect_error(.mask_id(sent_tok), regexp = "tokenizer do not encode <")
  # hf tokenizer
  # expect_error(.mask_id(hf_tok), regexp = "tokenizer do not encode <")
  # tokenizer.bpe
  expect_error(.mask_id(bpe_tok), regexp = "tokenizer do not encode <")

})

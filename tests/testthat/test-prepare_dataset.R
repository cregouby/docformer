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
  expect_error(tokenized <- .tokenize(tokenizer=sent_tok, phrase),NA)
  expect_true(purrr::vec_depth(tokenized)==2)
  expect_true(all)
  # hf tokenizer
  # expect_error(tokenized <- .tokenize(tokenizer=hf_tok, phrase),NA)
  # expect_true(purrr::vec_depth(tokenized)==2)
  # tokenizer.bpe
  expect_error(tokenized <- .tokenize(tokenizer=bpe_tok, phrase),NA)
  expect_true(purrr::vec_depth(tokenized)==2)

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
  expect_error(image_tt <- create_features_from_image(image, sent_tok_mask),
               regexp = NA)
  # class, shape and type
  expect_type(image_tt, "list")
  expect_s3_class(image_tt,"docformer_tensor")
  expect_equal(attr(image_tt, "max_seq_len"), 512L)
  expect_length(image_tt, 4)
  expect_equal(image_tt$x_features$shape, c(1, 512, 6))
  expect_equal(image_tt$y_features$shape, c(1, 512, 6))
  expect_equal(image_tt$text$shape, c(1, 512, 1))
  expect_equal(image_tt$image$shape[1:2], c(1, 3))
  expect_lte(image_tt$image$shape[3], 500)
  expect_lte(image_tt$image$shape[4], 384)

  # hf tokenizer
  # expect_error(.mask_id(hf_tok), regexp = "tokenizer do not encode <")

  # tokenizer.bpe
  expect_error(create_features_from_image(image, bpe_tok_mask),
               regexp = NA)
})

test_that("create_features_from_doc provides expected output from default values", {
  # Single-page document
  # sentencepiece
  qpdf::pdf_subset(doc, output = "2106.11539_1.pdf")
  expect_error(page1_tt <- create_features_from_doc("2106.11539_1.pdf", sent_tok_mask),
               regexp = NA)
  expect_type(page1_tt, "list")
  expect_s3_class(page1_tt,"docformer_tensor")
  expect_equal(attr(page1_tt, "max_seq_len"), 512L)
  expect_length(page1_tt, 4)
  expect_equal(page1_tt$x_features$shape, c(1, 512, 6))
  expect_equal(page1_tt$y_features$shape, c(1, 512, 6))
  expect_equal(page1_tt$text$shape, c(1, 512, 1))
  expect_equal(page1_tt$image$shape[1:2], c(1, 3))
  expect_lte(page1_tt$image$shape[3], 500)
  expect_lte(page1_tt$image$shape[4], 384)

  # hf tokenizer
  # expect_error(.mask_id(hf_tok), regexp = "tokenizer do not encode <")

  # tokenizer.bpe
  expect_error(create_features_from_doc("2106.11539_1.pdf", bpe_tok_mask),
               regexp = NA)

  # Multi-page document
  # sentencepiece
  expect_error(doc_tt <- create_features_from_doc(doc, sent_tok_mask),
               regexp = NA)
  expect_type(doc_tt, "list")
  expect_s3_class(doc_tt,"docformer_tensor")
  expect_equal(attr(doc_tt, "max_seq_len"), 512L)
  expect_length(doc_tt, 4)
  expect_equal(doc_tt$x_features$shape, c(2, 512, 6))
  expect_equal(doc_tt$y_features$shape, c(2, 512, 6))
  expect_equal(doc_tt$text$shape, c(2, 512, 1))
  expect_equal(doc_tt$image$shape[1:2], c(2, 3))
  expect_lte(doc_tt$image$shape[3], 500)
  expect_lte(doc_tt$image$shape[4], 384)

  # hf tokenizer
  # expect_error(.mask_id(hf_tok), regexp = "tokenizer do not encode <")

  # tokenizer.bpe
  expect_error(create_features_from_doc(doc, bpe_tok_mask),
               regexp = NA)
})

test_that("features properly save to disk and can be restored", {
  doc_tt <- create_features_from_doc(doc, sent_tok_mask)
  image_tt <- create_features_from_image(image, sent_tok_mask)
  withr::local_file({
    doc_file <- paste0(stringr::str_extract(doc, "[^/]+$"),".Rds")
    expect_error(save_featureRDS(doc_tt, file=doc_file),
                 regexp = NA)
    expect_true(file.exists(doc_file))
    expect_error(doc2_tt <- read_featureRDS(file=doc_file),
                 regexp = NA)
    expect_equal(purrr::map(doc2_tt,~.x$shape), purrr::map(doc_tt,~.x$shape))
    expect_equal(purrr::map_chr(doc2_tt,~.x$dtype %>% as.character),
                 purrr::map_chr(doc_tt,~.x$dtype %>% as.character))

    image_file <- paste0(stringr::str_extract(image, "[^/]+$"),".Rds")
    expect_error(save_featureRDS(image_tt, file=image_file),
                 regexp = NA)
    expect_true(file.exists(image_file))
    expect_error(image2_tt <- read_featureRDS(file=image_file),
                 regexp = NA)
    expect_equal(purrr::map(image2_tt,~.x$shape), purrr::map(image_tt,~.x$shape))
    expect_equal(purrr::map_chr(image2_tt,~.x$dtype %>% as.character),
                 purrr::map_chr(image_tt,~.x$dtype %>% as.character))
  })
})



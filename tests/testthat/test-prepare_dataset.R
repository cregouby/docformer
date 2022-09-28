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
  expect_no_error(tokenized <- .tokenize(tokenizer=sent_tok, phrase, 20))
  expect_true(purrr::vec_depth(tokenized)==2)
  # test of no unknown value, correct first value, correct  max_seq_len th value
  expect_true(all(purrr::map_lgl(tokenized %>% purrr::flatten(), ~.x>=1)))
  # hf tokenizer
  # expect_error(tokenized <- .tokenize(tokenizer=hf_tok, phrase),NA)
  # expect_true(purrr::vec_depth(tokenized)==2)
  # tokenizer.bpe
  expect_no_error(tokenized <- .tokenize(tokenizer=bpe_tok, phrase, 20))
  expect_true(purrr::vec_depth(tokenized)==2)
  # test of no unknown value, correct first value, correct max_seq_len th value
  expect_true(all(purrr::map_lgl(tokenized %>% purrr::flatten(), ~.x>=1)))
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
  expect_no_error(image_tt <- create_features_from_image(image, sent_tok_mask))
  # class, shape and type
  expect_type(image_tt, "list")
  expect_s3_class(image_tt,"docformer_tensor")
  expect_equal(attr(image_tt, "max_seq_len"), 512L)
  expect_length(image_tt, 5)
  expect_tensor_shape(image_tt$x_features, c(1, 512, 6))
  expect_tensor_shape(image_tt$y_features, c(1, 512, 6))
  expect_tensor_shape(image_tt$text, c(1, 512, 1))
  expect_tensor_shape(image_tt$mask, c(1, 512, 1))
  expect_tensor_dtype(image_tt$mask, "Bool")
  expect_tensor_dtype(image_tt$image, "Byte")
  # first and last tensors are separators
  expect_equal_to_r(image_tt$text[1, 1, 1], sent_tok_mask$vocabulary[sent_tok_mask$vocabulary$subword == "<s>",]$id)
  expect_equal_to_r(image_tt$text[1, 512, 1], sent_tok_mask$vocabulary[sent_tok_mask$vocabulary$subword == "</s>",]$id)
  expect_equal_to_tensor(image_tt$x_features[1, 1, 1], image_tt$x_features[1, 512, 1])

  expect_equal(image_tt$image$shape[1:2], c(1, 3))
  expect_lte(image_tt$image$shape[3], 500)
  expect_lte(image_tt$image$shape[4], 384)
  # values
  expect_true(all(image_tt$text %>% as.matrix >= 0))

  # hf tokenizer
  # expect_error(.mask_id(hf_tok), regexp = "tokenizer do not encode <")

  # tokenizer.bpe
  expect_no_error(create_features_from_image(image, bpe_tok_mask))
})

test_that("create_features_from_doc provides expected output from default values", {
  # Single-page document
  # sentencepiece
  qpdf::pdf_subset(doc, output = "2106.11539_1.pdf")
  expect_no_error(page1_tt <- create_features_from_doc("2106.11539_1.pdf", sent_tok_mask))
  expect_type(page1_tt, "list")
  expect_s3_class(page1_tt,"docformer_tensor")
  expect_equal(attr(page1_tt, "max_seq_len"), 512L)
  expect_length(page1_tt, 5)
  expect_tensor_shape(page1_tt$x_features, c(1, 512, 6))
  expect_tensor_shape(page1_tt$y_features, c(1, 512, 6))
  expect_tensor_shape(page1_tt$text, c(1, 512, 1))
  expect_tensor_shape(page1_tt$mask, c(1, 512, 1))
  expect_tensor_dtype(page1_tt$mask, "Bool")
  expect_tensor_dtype(page1_tt$image, "Byte")
  # first and last tensors are separators
  expect_equal_to_r(page1_tt$text[1, 1, 1], sent_tok_mask$vocabulary[sent_tok_mask$vocabulary$subword == "<s>",]$id)
  expect_equal_to_r(page1_tt$text[1, 512, 1], sent_tok_mask$vocabulary[sent_tok_mask$vocabulary$subword == "</s>",]$id)
  expect_equal_to_r(page1_tt$x_features[1,1,1], as.numeric(page1_tt$x_features[1,512,1]))
  # shape
  expect_equal(page1_tt$image$shape[1:2], c(1, 3))
  expect_true(all(tiny_tt$image$shape[3:4] <= c(500, 384)))
  # values
  expect_gte(page1_tt$text$min() %>% as.matrix, 0)

  # hf tokenizer
  # expect_error(.mask_id(hf_tok), regexp = "tokenizer do not encode <")

  # tokenizer.bpe
  expect_no_error(create_features_from_doc("2106.11539_1.pdf", bpe_tok_mask))

  # Multi-page document
  # sentencepiece
  expect_no_error(doc_tt <- create_features_from_doc(doc, sent_tok_mask))
  expect_type(doc_tt, "list")
  expect_s3_class(doc_tt,"docformer_tensor")
  expect_equal(attr(doc_tt, "max_seq_len"), 512L)
  expect_length(doc_tt, 5)
  expect_tensor_shape(doc_tt$x_features, c(2, 512, 6))
  expect_tensor_shape(doc_tt$y_features, c(2, 512, 6))
  expect_tensor_shape(doc_tt$text, c(2, 512, 1))
  expect_equal(doc_tt$image$shape[1:2], c(2, 3))
  expect_lte(doc_tt$image$shape[3], 500)
  expect_lte(doc_tt$image$shape[4], 384)
  expect_tensor_shape(doc_tt$mask, c(2, 512, 1))
  expect_tensor_dtype(doc_tt$mask, "Bool")
  expect_tensor_dtype(page1_tt$image, "Byte")
  # first and last tensors are separators
  expect_equal_to_r(page1_tt$text[1, 1, 1], sent_tok_mask$vocabulary[sent_tok_mask$vocabulary$subword == "<s>",]$id)
  expect_equal_to_r(page1_tt$text[1, 512, 1], sent_tok_mask$vocabulary[sent_tok_mask$vocabulary$subword == "</s>",]$id)
  expect_equal_to_r(page1_tt$x_features[1,1,1], as.numeric(page1_tt$x_features[1,512,1]))
  # shape
  expect_equal(page1_tt$image$shape[1:2], c(1, 3))
  expect_true(all(tiny_tt$image$shape[3:4] <= c(500, 384)))
  # values
  expect_gte(page1_tt$text$min() %>% as.matrix, 0)

  # hf tokenizer
  # expect_error(.mask_id(hf_tok), regexp = "tokenizer do not encode <")

  # tokenizer.bpe
  expect_no_error(create_features_from_doc("2106.11539_1.pdf", bpe_tok_mask))

  # Multi-page document
  # sentencepiece
  expect_no_error(doc_tt <- create_features_from_doc(doc, sent_tok_mask))
  expect_type(doc_tt, "list")
  expect_s3_class(doc_tt,"docformer_tensor")
  expect_equal(attr(doc_tt, "max_seq_len"), 512L)
  expect_length(doc_tt, 5)
  expect_tensor_shape(doc_tt$x_features, c(2, 512, 6))
  expect_tensor_shape(doc_tt$y_features, c(2, 512, 6))
  expect_tensor_shape(doc_tt$text, c(2, 512, 1))
  expect_equal(doc_tt$image$shape[1:2], c(2, 3))
  expect_lte(doc_tt$image$shape[3], 500)
  expect_lte(doc_tt$image$shape[4], 384)
  expect_tensor_shape(doc_tt$mask, c(2, 512, 1))
  expect_tensor_dtype(doc_tt$mask, "Bool")
  expect_tensor_dtype(doc_tt$image, "Byte")
  # values
  expect_gte(doc_tt$text$min() %>% as.numeric(), 0)

  # hf tokenizer
  # expect_error(.mask_id(hf_tok), regexp = "tokenizer do not encode <")

  # tokenizer.bpe
  expect_no_error(create_features_from_doc(doc, bpe_tok_mask))
})

test_that("create_features_from_doc correctly pads small content pages", {
  expect_no_error(doc_tt <- create_features_from_doc(doc2, sent_tok_mask))
  expect_type(doc_tt, "list")
  expect_s3_class(doc_tt,"docformer_tensor")
  expect_equal(attr(doc_tt, "max_seq_len"), 512L)
  expect_tensor_shape(doc_tt$x_features, c(2, 512, 6))
  expect_tensor_shape(doc_tt$y_features, c(2, 512, 6))
  expect_tensor_shape(doc_tt$text, c(2, 512, 1))
  expect_tensor_shape(doc_tt$mask, c(2, 512, 1))
  expect_tensor_dtype(doc_tt$mask, "Bool")
  expect_tensor_dtype(doc_tt$image, "Byte")
  # values
  expect_gte(doc_tt$text$min() %>% as.numeric, 0)

  # hf tokenizer
  # expect_error(.mask_id(hf_tok), regexp = "tokenizer do not encode <")

  # tokenizer.bpe
  expect_no_error(create_features_from_doc(doc2, bpe_tok_mask))
})

test_that("there is no masking on padded text", {
  doc_tt <- create_features_from_doc(doc2, sent_tok_mask)
  padded_text <- torch::torch_cat(list(doc_tt$text[1,,],doc_tt$mask[1,,]),dim = 2)
  pad_mask <- torch::torch_unique_consecutive(padded_text, dim = 1, return_counts = TRUE)
  expect_equal_to_r(torch::torch_sum(pad_mask[[1]][, 1] == 1, dim = 1), 1)
})

test_that("create_features_from_* correctly manages image with small target_geometry", {
  # _from_doc
  expect_no_error(tiny_tt <- create_features_from_doc(doc, sent_tok_mask, target_geometry = "128x168"))
  expect_equal(tiny_tt$image$shape[1:2], c(2, 3))
  expect_true(all(tiny_tt$image$shape[3:4] <= c(168, 128)))
  expect_lte(tiny_tt$x_features$max() %>% as.numeric, 128)
  expect_lte(tiny_tt$y_features$max() %>% as.numeric, 168)

  # _from_image
  expect_no_error(tiny_tt <- create_features_from_image(image, bpe_tok_mask, target_geometry = "128x168"))
  expect_equal(tiny_tt$image$shape[1:2], c(1, 3))
  expect_true(all(tiny_tt$image$shape[3:4] <= c(168, 128)))
  expect_lte(tiny_tt$x_features$max() %>% as.numeric, 128)
  expect_lte(tiny_tt$y_features$max() %>% as.numeric, 168)

  # _from_docbank
  expect_no_error(tiny_tt <- create_features_from_docbank(docbank_txt, docbank_img,sent_tok_mask, target_geometry = "128x168"))
  expect_equal(tiny_tt$image$shape[1:2], c(1, 3))
  expect_true(all(tiny_tt$image$shape[3:4] <= c(168, 128)))
  expect_lte(tiny_tt$x_features$max() %>% as.numeric, 128)
  expect_lte(tiny_tt$y_features$max() %>% as.numeric, 168)
})

test_that("features properly save to disk and can be restored", {
  doc_tt <- create_features_from_doc(doc, sent_tok_mask)
  image_tt <- create_features_from_image(image, sent_tok_mask)
  withr::local_file({
    doc_file <- paste0(stringr::str_extract(doc, "[^/]+$"),".Rds")
    expect_no_error(save_featureRDS(doc_tt, file = doc_file))
    expect_true(file.exists(doc_file))
    expect_no_error(doc2_tt <- read_featureRDS(file = doc_file))
    expect_equal(purrr::map(doc2_tt,~.x$shape), purrr::map(doc_tt,~.x$shape))
    expect_equal(purrr::map_chr(doc2_tt,~.x$dtype %>% as.character),
                 purrr::map_chr(doc_tt,~.x$dtype %>% as.character))

    image_file <- paste0(stringr::str_extract(image, "[^/]+$"),".Rds")
    expect_no_error(save_featureRDS(image_tt, file = image_file))
    expect_true(file.exists(image_file))
    expect_no_error(image2_tt <- read_featureRDS(file = image_file))
    expect_equal(purrr::map(image2_tt,~.x$shape), purrr::map(image_tt,~.x$shape))
    expect_equal(purrr::map_chr(image2_tt,~.x$dtype %>% as.character),
                 purrr::map_chr(image_tt,~.x$dtype %>% as.character))
  })
})

test_that("mask_for_mm_mlm works as expected", {
  expect_no_error(masked_doc <- mask_for_mm_mlm(doc_tt, .mask_id(sent_tok_mask)))
  expect_tensor_shape(masked_doc$text, c(2, 512, 1))
  # expect_equal_to_r() would be too complex here

  all_masked_tt <- doc_tt
  all_masked_tt$mask <- torch_zeros_like(doc_tt$mask)
  masked_doc <- mask_for_mm_mlm(all_masked_tt, 99999)
  expect_equal_to_r(masked_doc$text, array(rep(99999, prod(masked_doc$text$shape)), dim = masked_doc$text$shape))
})

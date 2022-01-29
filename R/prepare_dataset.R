#' Normalize a bounding-box
#'
#' Takes one or more bounding box and normalize each of their dimensions to `size`. If you notice it is
#' just like calculating percentage except takes `size=1000` instead of 100.
#'
#' @param box (vector of 4 int): the original bounding-box coordinates (xmin, ymin, xmax, ymax)
#' @param width (int): total width of the image
#' @param height (int): total height of the image
#' @param size (int): target value to normalize bounding_box to. (default 1000)
#'
#' @return a vector of size 4 integers with values normalised to `size`.
#' @export
#'
#' @examples
#' # normalise bounding-box in percent
#' normalize_box(c(227,34,274,41), width=2100, height=3970, size=100)
normalize_box <- function(bbox, width, height, size=1000) {
  norm_bbox <- round(c(bbox[[1]]/width, bbox[[2]]/height, bbox[[3]]/width, bbox[[4]]/height) * size)
  return(norm_bbox)
}

resize_align_bbox <- function(bbox, origin_w, origin_h, target_w, target_h) {
  res_bbox <- round(c(bbox[[1]]*target_w / orig_w, bbox[[2]]*target_h / orig_h,
                      bbox[[3]]*target_w / orig_w, bbox[[4]]*target_h / orig_h))
  return(res_bbox)
}

#' Apply tesseract::ocr_data() and clean result.
#'
#' @param image file path, url, or raw vector to image (png, tiff, jpeg, etc)
#'
#' @return a data.frame of words and associated bounding-box
#' @export
#'
#' @examples
#' # good quality scan
#' df <- apply_ocr(image1)
#' # poor quality scan
#' df <- image2 %>% image_read() %>%
#'    image_resize("2000x") %>%
#'    image_trim(fuzz=40) %>%
#'    image_write(format='png', density="300x300") %>%
#'    apply_ocr()
apply_ocr <- function(image) {
  ocr_df <- tesseract::ocr_data(image) %>%
    mutate(poor_word =( stringr::str_detect(word, "^\\W+$|\\W{3,}") | confidence <20),
           bb = bbox %>% stringr::str_split(",") %>% purrr::map(~readr::parse_number(.x))) %>%
    tidyr::unnest_wider(bb, names_sep="_") %>%
    filter(!poor_word) %>%
    select(word, confidence, xmin=bb_1, ymin=bb_2, xmax=bb_3, ymax=bb_4)
  return(ocr_df)
}

#' tokenize the character vector and prepend the \[CLS\] token to first
#'
#' @param tokenizer the tokenizer function
#' @param x character vector to encode
#' @param ...  unused
#'
#' @return list of token ids for each token
.tokenize <- function(tokenizer, ...) {
  UseMethod(".tokenize")
}
.tokenize.default <- function(tokenizer, x ) {
  rlang::abort(paste0(tokenizer," is not recognized as a supported tokenizer"))
}
.tokenize.tokenizer <- function(tokenizer, x) {
  idx <- purrr::map(x,~tokenizer$encode(.x)$ids)
  idx[[1]] <- idx[[1]] %>% purrr::prepend(tokenizer$encode("[CLS]")$ids)
  return(idx)
}
.tokenize.youtokentome <- function(tokenizer, x) {
  idx <- purrr::map(x,~tokenizers.bpe::bpe_encode(tokenizer, .x, type="ids"))
  idx[[1]] <- idx[[1]] %>%
    purrr::prepend(tokenizers.bpe::bpe_encode(tokenizer,"<BOS>", type="ids"))
  return(idx)
}
.tokenize.sentencepiece <- function(tokenizer, x) {
  idx <- purrr::map(x,~sentencepiece::sentencepiece_encode(tokenizer, .x, type="ids"))
  idx[[1]] <- idx[[1]] %>%
    purrr::prepend(sentencepiece::sentencepiece_encode(tokenizer, "<s>", type="ids"))
  # see https://github.com/google/sentencepiece/blob/master/doc/special_symbols.md for <mask>
  return(idx)
}

.mask_id <- function(tokenizer) {
  UseMethod(".mask_id")
}
.mask_id.default <- function(tokenizer ) {
  rlang::abort(paste0(tokenizer," is not recognized as a supported tokenizer"))
}
.mask_id.tokenizer <- function(tokenizer) {
  return(tokenizer$encode("[MASK]")$ids)
}
.mask_id.youtokentome <- function(tokenizer) {
  return(tokenizers.bpe::bpe_encode(tokenizer,"<MASK>", type="ids"))
}
.mask_id.sentencepiece <- function(tokenizer) {
  # see https://github.com/google/sentencepiece/blob/master/doc/special_symbols.md for <mask>
  return(sentencepiece::sentencepiece_encode(tokenizer, "<mask>", type="ids"))
}

#' Turn image into docformer torch tensor input feature
#'
#' @param image file path, url, or raw vector to image (png, tiff, jpeg, etc)
#' @param tokenizer tokenizer function to apply to words extracted from image. Currently,
#'   {hftokenizers}, {tokenizer.bpe} and {sentencepiece} tokenizer are supported.
#' @param add_batch_dim (boolean) add a extra dimension to tensor for batch encoding
#' @param target_geometry image target magik geometry expected by the image model input
#' @param max_seq_len size of the embedding vector in tokens
#' @param save_to_disk (boolean) shall we save the result onto disk
#' @param path_to_save result path
#' @param apply_mask_for_mlm add mask to the language model
#' @param debugging additionnal feature for debugging purposes
#'
#' @return
#' @export
#'
#' @examples
create_features_from_image <- function(image,
                            tokenizer,
                            add_batch_dim=TRUE,
                            target_geometry="224x224",
                            max_seq_len=512,
                            save_to_disk=FALSE,
                            path_to_save=NULL,
                            apply_mask_for_mlm=FALSE,
                            debugging=FALSE) {

  # step 0 prepare utilities datasets
  mask_for_mlm <- if (apply_mask_for_mlm) {
    runif(n=min(nrow(encoding), max_seq_len))>0.15
  } else {
    rep(TRUE, max_seq_len)
  }
  mask_id <- .mask_id(tokenizer)
  empty_encoding <- dplyr::tibble(xmin = rep(0,max_seq_len),
                           ymin = rep(0,max_seq_len),
                           xmax = rep(0,max_seq_len),
                           ymax = rep(0,max_seq_len),
                           word = NA_character_,
                           idx = list(0),
                           prior=TRUE)

  # step 1 read images and its attributes
  original_image <- magick::image_read(image)
  w_h <- magick::image_info(original_image)

  # step 2: resize image
  resized_image <- magick::image_resize(original_image, geometry=target_geometry)

  # step 3 extract text throuhg OCR and normalize bbox to 1000x1000
  encoding <- apply_ocr(original_image) %>%
    dplyr::mutate(xmin = round(xmin/w_h$width * 1000),
           ymin= round(ymin/w_h$height * 1000),
           xmax = round(xmax/w_h$width * 1000),
           ymax= round(ymax/w_h$height * 1000),
    # step 4 tokenize words into `idx` and get their bbox
           idx = .tokenize(tokenizer, word)) %>%
    dplyr::select(-confidence) %>%
    # step 5.1 apply mask for the sake of pre-training
    dplyr::bind_cols(prior=mask_for_mlm) %>%
    # step 5.2: fill in a max_seq_len matrix
    dplyr::bind_rows(empty_encoding)

  encoding_long <- encoding  %>%
    tidyr::unnest_longer(col="idx") %>%
    # step 5.3: truncate seq. to maximum length
    dplyr::slice_head(n=max_seq_len) %>%
    # step 6: (nill here)
    # step 7: apply mask for the sake of pre-training
    mutate(idx = ifelse(prior, idx, mask_id))
    # step 8 normlize the image

  encoding_tt <- torch::torch_stack(
    encoding = torch::torch_tensor(encoding)
  )

}
#' Turn document into docformer torch tensor input feature
#'
#' @param doc file path, url, or raw vector to document (currently pdf only)
#' @param tokenizer tokenizer function to apply to words extracted from image. Currently,
#'   {hftokenizers}, {tokenizer.bpe} and {sentencepiece} tokenizer are supported.
#' @param add_batch_dim (boolean) add a extra dimension to tensor for batch encoding
#' @param target_geometry image target magik geometry expected by the image model input
#' @param max_seq_len size of the embedding vector in tokens
#' @param save_to_disk (boolean) shall we save the result onto disk
#' @param path_to_save result path
#' @param apply_mask_for_mlm add mask to the language model
#' @param extras_for_debugging additionnal feature for debugging purposes
#'
#' @return
#' @export
#'
#' @examples
create_features_from_doc <- function(doc,
                                        tokenizer,
                                        add_batch_dim=TRUE,
                                        target_geometry="224x224",
                                        max_seq_len=512,
                                        save_to_disk=FALSE,
                                        path_to_save=NULL,
                                        apply_mask_for_mlm=FALSE,
                                        extras_for_debugging=FALSE) {
}

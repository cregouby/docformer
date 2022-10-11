#' Normalize a bounding-box
#'
#' Takes one or more bounding box and normalize each of their dimensions to `size`. If you notice it is
#' just like calculating percentage except takes `size = 1000` instead of 100.
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
#' normalize_box(c(227,34,274,41), width = 2100, height = 3970, size = 100)
normalize_box <- function(bbox, width, height, size = 1000) {
  norm_bbox <- trunc(c(bbox[[1]] / width, bbox[[2]] / height, bbox[[3]] / width, bbox[[4]] / height) * size)
  return(norm_bbox)
}

resize_align_bbox <- function(bbox, origin_w, origin_h, target_w, target_h) {
  res_bbox <- trunc(c(bbox[[1]] * target_w / origin_w, bbox[[2]] * target_h / origin_h,
                      bbox[[3]] * target_w / origin_w, bbox[[4]] * target_h / origin_h))
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
#'    image_trim(fuzz = 40) %>%
#'    image_write(format = 'png', density = "300x300") %>%
#'    apply_ocr()
apply_ocr <- function(image) {
  ocr_df <- tesseract::ocr_data(image) %>%
    dplyr::mutate(poor_word = (stringr::str_detect(word, "^\\W+$|\\W{3,}") | confidence < 20),
                  bb = bbox %>% stringr::str_split(",") %>% purrr::map(as.integer)) %>%
    tidyr::unnest_wider(bb, names_sep = "_") %>%
    dplyr::filter(!poor_word) %>%
    dplyr::select(text = word, confidence, xmin = bb_1, ymin = bb_2, xmax = bb_3, ymax = bb_4)
  return(ocr_df)
}

#' tokenize the character vector and prepend the \[CLS\] token to first
#'
#' @param tokenizer the tokenizer function
#' @param x character vector to encode
#' @param max_seq_len  unused
#' @export
#' @return list of token ids for each token
.tokenize <- function(tokenizer, x, max_seq_len) {
  UseMethod(".tokenize")
}
#' @export
.tokenize.default <- function(tokenizer, x, max_seq_len) {
  rlang::abort(paste0(tokenizer, " is not recognized as a supported tokenizer"))
}
#' @export
.tokenize.tokenizer <- function(tokenizer, x) {
  idx <- purrr::map(x, ~tokenizer$encode(.x)$ids)
  # TODO BUG shall shift-right after max_seq_len slicing
  # idx[[1]] <- idx[[1]] %>% purrr::prepend(tokenizer$encode("[CLS]")$ids)
  return(idx)
}
#' @export
.tokenize.youtokentome <- function(tokenizer, x, max_seq_len) {
  idx <- purrr::map(x, ~tokenizers.bpe::bpe_encode(tokenizer, .x, type = "ids")[[1]])
  # # prepend sequence with CLS token
  # idx[[1]] <- dplyr::first(idx) %>%
  #   purrr::prepend(tokenizer$vocabulary[tokenizer$vocabulary$subword =  = "<BOS>",]$id %>% as.integer)
  # # append SEP token at max_seq_len position
  # cum_idx <- cumsum(purrr::map_dbl(idx, length))
  # max_seq_idx <- min(dplyr::last(which(cum_idx<max_seq_len))+1, length(idx))
  # pre_sep_position <- max(0,max_seq_len - cum_idx[max_seq_idx-1] - 1)
  # idx[[max_seq_idx]] <- idx[[max_seq_idx]] %>%
  #   append(tokenizer$vocabulary[tokenizer$vocabulary$subword =  = "<EOS>",]$id %>% as.integer, after = pre_sep_position)
  return(idx)
}
#' @export
.tokenize.sentencepiece <- function(tokenizer, x, max_seq_len) {
  idx <- purrr::map(x, ~sentencepiece::sentencepiece_encode(tokenizer, .x, type = "ids")[[1]])
  # # prepend sequence with CLS token
  # idx[[1]] <- dplyr::first(idx) %>%
  #   purrr::prepend(tokenizer$vocabulary[tokenizer$vocabulary$subword =  = "<s>",]$id %>% as.integer)
  # # append SEP token at max_seq_len position
  # cum_idx <- cumsum(purrr::map_dbl(idx, length))
  # max_seq_idx <- min(dplyr::last(which(cum_idx<max_seq_len))+1, length(idx))
  # pre_sep_position <- max(0,max_seq_len - cum_idx[max_seq_idx-1] - 1 )
  # idx[[max_seq_idx]] <- idx[[max_seq_idx]] %>%
  #   append(tokenizer$vocabulary[tokenizer$vocabulary$subword =  = "</s>",]$id %>% as.integer, after = pre_sep_position)
  # # see https://github.com/google/sentencepiece/blob/master/doc/special_symbols.md for <mask>
  return(idx)
}

#' @export
#' @rdname special_tokens
.mask_id <- function(tokenizer) {
  UseMethod(".mask_id")
}
#' @export
.mask_id.default <- function(tokenizer ) {
  rlang::abort(paste0(tokenizer, " is not recognized as a supported tokenizer"))
}
#' @export
.mask_id.tokenizer <- function(tokenizer) {
  mask_id <- tokenizer$encode("[MASK]")$ids
  if (length(mask_id) == 0) {
    rlang::abort("tokenizer do not encode `[MASK]` properly.")
  }
  return(mask_id)
}
#' @export
.mask_id.youtokentome <- function(tokenizer) {
  mask_id <- tokenizer$vocabulary[tokenizer$vocabulary$subword == "<MASK>",]$id
  if (length(mask_id) == 0) {
    rlang::abort("tokenizer do not encode `<MASK>` properly.")
  }
  return(mask_id)
}
#' @export
.mask_id.sentencepiece <- function(tokenizer) {
  # see https://github.com/google/sentencepiece/blob/master/doc/special_symbols.md for <mask>
  mask_id <- tokenizer$vocabulary[tokenizer$vocabulary$subword == "<mask>",]$id
  if (length(mask_id) == 0) {
    rlang::abort("tokenizer do not encode `<mask>` properly.")
  }
  return(mask_id)
}

#' Extract special tokens from tokenizer
#'
#' @export
#' @rdname special_tokens
.pad_id <- function(tokenizer) {
  UseMethod(".pad_id")
}
#' @export
.pad_id.default <- function(tokenizer ) {
  rlang::abort(paste0(tokenizer, " is not recognized as a supported tokenizer"))
}
#' @export
.pad_id.tokenizer <- function(tokenizer) {
  pad_id <- tokenizer$encode("[PAD]")$ids
  if (length(pad_id) == 0) {
    rlang::abort("tokenizer do not encode `[PAD]` properly.")
  }
  return(pad_id)
}
#' @export
.pad_id.youtokentome <- function(tokenizer) {
  pad_id <- tokenizer$vocabulary[tokenizer$vocabulary$subword == "<PAD>",]$id
  if (length(pad_id) == 0) {
    rlang::abort("tokenizer do not encode `<PAD>` properly.")
  }
  return(pad_id)
}
#' @export
.pad_id.sentencepiece <- function(tokenizer) {
  # see https://github.com/google/sentencepiece/blob/master/doc/special_symbols.md for <mask>
  pad_id <- tokenizer$vocabulary[tokenizer$vocabulary$subword == "<pad>",]$id
  if (length(pad_id) == 0) {
    rlang::abort("tokenizer do not encode `<pad>` properly.")
  }
  return(pad_id)
}

#' @export
#' @rdname special_tokens
.sep_id <- function(tokenizer) {
  UseMethod(".sep_id")
}
#' @export
.sep_id.default <- function(tokenizer ) {
  rlang::abort(paste0(tokenizer," is not recognized as a supported tokenizer"))
}
#' @export
.sep_id.tokenizer <- function(tokenizer) {
  sep_id <- tokenizer$encode("[SEP]")$ids
  if (length(sep_id) == 0) {
    rlang::abort("tokenizer do not encode `[SEP]` properly.")
  }
  return(sep_id)
}
#' @export
.sep_id.youtokentome <- function(tokenizer) {
  sep_id <- tokenizer$vocabulary[tokenizer$vocabulary$subword == "<EOS>",]$id
  if (length(sep_id) == 0) {
    rlang::abort("tokenizer do not encode `<EOS>` properly.")
  }
  return(sep_id)
}
#' @export
.sep_id.sentencepiece <- function(tokenizer) {
  # see https://github.com/google/sentencepiece/blob/master/doc/special_symbols.md for <mask>
  sep_id <- tokenizer$vocabulary[tokenizer$vocabulary$subword == "</s>",]$id
  if (length(sep_id) == 0) {
    rlang::abort("tokenizer do not encode `</s>` properly.")
  }
  return(sep_id)
}

#' @export
#' @rdname special_tokens
.cls_id <- function(tokenizer) {
  UseMethod(".cls_id")
}
#' @export
.cls_id.default <- function(tokenizer ) {
  rlang::abort(paste0(tokenizer," is not recognized as a supported tokenizer"))
}
#' @export
.cls_id.tokenizer <- function(tokenizer) {
  cls_id <- tokenizer$encode("[CLS]")$ids
  if (length(cls_id) == 0) {
    rlang::abort("tokenizer do not encode `[CLS]` properly.")
  }
  return(cls_id)
}
#' @export
.cls_id.youtokentome <- function(tokenizer) {
  cls_id <- tokenizer$vocabulary[tokenizer$vocabulary$subword == "<BOS>",]$id
  if (length(cls_id) == 0) {
    rlang::abort("tokenizer do not encode `<BOS>` properly.")
  }
  return(cls_id)
}
#' @export
.cls_id.sentencepiece <- function(tokenizer) {
  # see https://github.com/google/sentencepiece/blob/master/doc/special_symbols.md for <mask>
  cls_id <- tokenizer$vocabulary[tokenizer$vocabulary$subword == "<s>", ]$id
  if (length(cls_id) == 0) {
    rlang::abort("tokenizer do not encode `<s>` properly.")
  }
  return(cls_id)
}

.padding_encode <- function(max_seq_len, pad_id) {
  dplyr::tibble(xmin = rep(0, max_seq_len),
                ymin = rep(0, max_seq_len),
                xmax = rep(0, max_seq_len),
                ymax = rep(0, max_seq_len),
                x_width = rep(0, max_seq_len),
                y_height = rep(0, max_seq_len),
                x_min_d = rep(0, max_seq_len),
                y_min_d = rep(0, max_seq_len),
                x_max_d = rep(0, max_seq_len),
                y_max_d = rep(0, max_seq_len),
                x_center_d = rep(0, max_seq_len),
                y_center_d = rep(0, max_seq_len),
                text = NA_character_,
                idx = pad_id,
                mlm_mask = TRUE
                )

}

create_feature <- function(filepath, config) {
  if (fs::is_dir(filepath)) {
    filepath <- list.files(filepath)
  }
  # check if tokenizer url exist
  tok_url <- transformers_config[transformers_config$model_name == config$pretrained_model_name, ]$tokenizer_json
  stopifnot("Tokenizer url cannot be found for model from config file" = length(tok_url) > 0)

  # initialize tokenizer
  tok_json <- jsonlite::stream_in(url(tok_url))
  tok_pkg <- dplyr::case_when((tok_json$model$type %||% tok_json$decoder$type) == "BPE" ~ "tokenizers.bpe",
                              (tok_json$model$type %||% tok_json$decoder$type) == "WordPiece" ~ "sentencepiece",
                              TRUE ~ "Unknown")
  tok_tmp <- tempfile(fileext = ".json")
  jsonlite::stream_out(tok_json, file(tok_tmp))
  tokenizer <- dplyr::case_when(tok_pkg == "tokenizers.bpe" ~ tokenizers.bpe::bpe_load_model(tok_tmp),
                                tok_pkg == "sentencepiece" ~ sentencepiece::sentencepiece_load_model(tok_tmp))
  # check if tokenizer is compatible with model
  stopifnot("Tokenizer vocabulary size is not compatible with the one from model config file" = tokenizer$vocab_size <= config$vocab_size)


  # dispatch files according to their extension

  # coro loop on files
}
#' Turn image into docformer torch tensor input feature
#'
#' @param image file path, url, or raw vector to image (png, tiff, jpeg, etc)
#' @param tokenizer tokenizer function to apply to words extracted from image. Currently,
#'   {hftokenizers}, {tokenizer.bpe} and {sentencepiece} tokenizer are supported.
#' @param add_batch_dim (boolean) add a extra dimension to tensor for batch encoding
#' @param target_geometry image target magik geometry expected by the image model input
#' @param max_seq_len size of the embedding vector in tokens
#' @param debugging additionnal feature for debugging purposes
#'
#' @return a list of named tensors
#' @export
#'
#' @examples
#' # load a tokenizer with <mask> encoding capability
#' sent_tok <- sentencepiece::sentencepiece_load_model(
#'   system.file(package = "sentencepiece", "models/nl-fr-dekamer.model")
#' )
#' sent_tok$vocab_size <- sent_tok$vocab_size+1L
#' sent_tok$vocabulary <- rbind(
#'   sent_tok$vocabulary,
#'   data.frame(id = sent_tok$vocab_size, subword = "<mask>")
#' )
#' # turn pdf into feature
#' image <- system.file(package = "docformer", "inst", "2106.11539_1.png")
#' image_tt <- create_features_from_image(image, tokenizer = sent_tok)
#'
create_features_from_image <- function(image,
                                       tokenizer,
                                       add_batch_dim = TRUE,
                                       target_geometry = "384x500",
                                       max_seq_len = 512,
                                       debugging = FALSE) {

  # step 0 prepare utilities datasets
  # mask_id <- .mask_id(tokenizer)
  pad_id <- .pad_id(tokenizer)
  # step 1 read images and its attributes
  original_image <- magick::image_read(image)
  w_h <- magick::image_info(original_image)
  target_w_h <- stringr::str_split(target_geometry, "x")[[1]] %>%
    as.numeric()
  scale_w <- target_w_h[1] /  w_h$width
  scale_h <- target_w_h[2] / w_h$height
  CLS_TOKEN_BOX_long <- c(idx = .cls_id(tokenizer), xmax = target_w_h[1], x_width = target_w_h[1], ymax = target_w_h[2], y_height = target_w_h[2],
                          xmin = 0, ymin = 0, x_min_d = 0, x_max_d = 0, x_center_d = 0, y_min_d = 0, y_max_d = 0, y_center_d = 0)
  SEP_TOKEN_BOX_long <- c(idx = .sep_id(tokenizer), xmax = target_w_h[1], x_width = target_w_h[1], ymax = target_w_h[2], y_height = target_w_h[2],
                          xmin = 0, ymin = 0, x_min_d = 0, x_max_d = 0, x_center_d = 0, y_min_d = 0, y_max_d = 0, y_center_d = 0)

  # step 3 extract text throuhg OCR and normalize bbox to target geometry
  encoding <- apply_ocr(original_image) %>%
    dplyr::mutate(
      # step 10 normalize the bbox
      xmin = trunc(xmin * scale_w),
      ymin = trunc(ymin * scale_h),
      xmax = trunc(xmax * scale_w),
      ymax = trunc(ymax * scale_h),
      x_center = trunc((xmin + xmax )/2),
      y_center = trunc((ymin + ymax )/2),
      # step 11 add relative spatial features
      x_width = xmax - xmin,
      y_height = ymax - ymin,
      x_min_d = dplyr::lead(xmin) - xmin,
      y_min_d = dplyr::lead(ymin) - ymin,
      x_max_d = dplyr::lead(xmax) - xmin,
      y_max_d = dplyr::lead(ymax) - ymin,
      x_center_d = dplyr::lead(x_center) - x_center,
      y_center_d = dplyr::lead(y_center) - y_center,
      # step 4 tokenize words into `idx` and get their bbox
      idx = .tokenize(tokenizer, text, max_seq_len)) %>%
    dplyr::select(-confidence, -x_center, -y_center) %>%
    tidyr::replace_na(list("", rep(0, 13)))

  encoding_long <- encoding  %>%
    # step 5: apply mask for the sake of pre-training
    dplyr::mutate(mlm_mask = stats::runif(n = nrow(encoding) ) > 0.15) %>%
    # step 6: unnest tokens
    tidyr::unnest_longer(col = "idx") %>%
    # step 7: truncate seq. to maximum length - 2
    dplyr::slice_head(n = max_seq_len - 2) %>%
    # step 8, 9, 10: prepend sequence with CLS token then append SEP token at last position, then pad to max_seq_len
    dplyr::bind_rows(CLS_TOKEN_BOX_long, ., SEP_TOKEN_BOX_long, .padding_encode(max_seq_len, pad_id)) %>%
    # step 11: truncate seq. to maximum length
    dplyr::slice_head(n = max_seq_len)

  # step 12 convert all to tensors
  # x_feature, we keep xmin, xmax, x_width, x_min_d, x_max_d, x_center_d
  x_features <-  encoding_long %>% dplyr::select(xmin, xmax, x_width, x_min_d, x_max_d, x_center_d) %>%
    as.matrix %>% torch::torch_tensor(dtype = torch::torch_int())
  # y_feature
  y_features <- encoding_long %>% dplyr::select(ymin, ymax, y_height, y_min_d, y_max_d, y_center_d) %>%
    as.matrix %>% torch::torch_tensor(dtype = torch::torch_int())
  # text (used to be input_ids)
  text <- encoding_long %>% dplyr::select(idx) %>%
    as.matrix %>% torch::torch_tensor(dtype = torch::torch_int())
  # image
  image <- original_image %>% torchvision::transform_resize(size = target_geometry) %>%
    torchvision::transform_to_tensor() * 255
  # masks
  mask <- encoding_long %>% dplyr::select(mlm_mask) %>% tidyr::replace_na(list(mlm_mask = TRUE)) %>%
    as.matrix %>% torch::torch_tensor(dtype = torch::torch_bool())
  # step 13: add tokens for debugging

  # step 14: add extra dim for batch
  encoding_lst <- if (add_batch_dim) {
    list(x_features = x_features$unsqueeze(1), y_features = y_features$unsqueeze(1), text = text$unsqueeze(1), image = image$to(dtype = torch::torch_uint8())$unsqueeze(1), mask = mask$unsqueeze(1))
  } else {
    list(x_features = x_features, y_features = y_features, text = text, image = image$to(dtype = torch::torch_uint8()), mask = mask)
  }
  # step 16: void keys to keep, resized_and_al&igned_bounding_boxes have been added for the purpose
  # to test if the bounding boxes are drawn correctly or not, it maybe removed
  class(encoding_lst) <- "docformer_tensor"
  attr(encoding_lst, "max_seq_len") <- max_seq_len
  encoding_lst

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
#' @param extras_for_debugging additionnal feature for debugging purposes
#'
#' @return a list of named tensors
#' @export
#'
#' @examples
#' # load a tokenizer with <mask> encoding capability
#' sent_tok <- sentencepiece::sentencepiece_load_model(
#'    system.file(package = "sentencepiece", "models/nl-fr-dekamer.model")
#'    )
#' sent_tok$vocab_size <- sent_tok$vocab_size+1L
#' sent_tok$vocabulary <- rbind(
#'   sent_tok$vocabulary,
#'   data.frame(id = sent_tok$vocab_size, subword = "<mask>")
#'   )
#' # turn pdf into feature
#' doc <- system.file(package = "docformer", "inst", "2106.11539_1_2.pdf")
#' doc_tt <- create_features_from_doc(doc, tokenizer = sent_tok)
#'
create_features_from_doc <- function(doc,
                                     tokenizer,
                                     add_batch_dim = TRUE,
                                     target_geometry = "384x500",
                                     max_seq_len = 512,
                                     extras_for_debugging = FALSE) {
  # step 0 prepare utilities datasets
  # mask_id <- .mask_id(tokenizer)
  pad_id <- .pad_id(tokenizer)
  # step 1 read document and its attributes
  # TODO improvement: use the actual text boundaries for finer text accuracy
  w_h <- pdftools::pdf_pagesize(doc)
  target_w_h <- stringr::str_split(target_geometry, "x")[[1]] %>%
    as.numeric()
  scale_w <- target_w_h[1] / w_h$width
  scale_h <- target_w_h[2] / w_h$height
  # TODO improvement : accept variable CLS_TOKEN_BOX  as it an be variable, but as per the paper,
  # they have mentioned that it covers the whole image. Like:
  # CLS_TOKEN_BOX <- bind_rows(xmin = 0, ymin = 0, x_width = w_h$width, y_height = w_h$height)
  CLS_TOKEN_BOX_long <- c(idx = .cls_id(tokenizer), xmax = target_w_h[1], x_width = target_w_h[1], ymax = target_w_h[2], y_height = target_w_h[2],
                          xmin = 0, ymin = 0, x_min_d = 0, x_max_d = 0, x_center_d = 0, y_min_d = 0, y_max_d = 0, y_center_d = 0)
  SEP_TOKEN_BOX_long <- c(idx = .sep_id(tokenizer), xmax = target_w_h[1], x_width = target_w_h[1], ymax = target_w_h[2], y_height = target_w_h[2],
                          xmin = 0, ymin = 0, x_min_d = 0, x_max_d = 0, x_center_d = 0, y_min_d = 0, y_max_d = 0, y_center_d = 0)

  # step 3 extract text
  encoding <-  purrr::pmap(list(pdftools::pdf_data(doc), as.list(scale_w), as.list(scale_h)),
                           ~..1 %>% dplyr::mutate(
                             # step 10 normalize the bbox
                             xmin = trunc( x * ..2),
                             ymin = trunc( y * ..3),
                             xmax = trunc((x + width) * ..2),
                             ymax = trunc((y + height) * ..3),
                             x_center = trunc((xmin + xmax )/2),
                             y_center = trunc((ymin + ymax )/2),
                             # step 11 add relative spatial features
                             x_width = xmax - xmin,
                             y_height = ymax - ymin,
                             x_min_d = dplyr::lead(xmin) - xmin,
                             y_min_d = dplyr::lead(ymin) - ymin,
                             x_max_d = dplyr::lead(xmax) - xmin,
                             y_max_d = dplyr::lead(ymax) - ymin,
                             x_center_d = dplyr::lead(x_center) - x_center,
                             y_center_d = dplyr::lead(y_center) - y_center,
                             # step 4 tokenize words into `idx` and get their bbox
                             idx = .tokenize(tokenizer, text, max_seq_len)) %>%
                             dplyr::select(-x_center, -y_center) %>%
                             tidyr::replace_na(list("", rep(0, 13))))

  encoding_long <- purrr::map(encoding, ~.x  %>%
                                # step 5: apply mask for the sake of pre-training
                                dplyr::mutate(mlm_mask = stats::runif(n = nrow(.x)) > 0.15) %>%
                                # step 6: unnest tokens
                                tidyr::unnest_longer(col = "idx") %>%
                                # step 7: truncate seq. to maximum length - 2
                                dplyr::slice_head(n = max_seq_len-2) %>%
                                # step 8, 9, 10: prepend sequence with CLS token then append SEP token at last position, then pad to max_seq_len
                                dplyr::bind_rows(CLS_TOKEN_BOX_long, ., SEP_TOKEN_BOX_long, .padding_encode(max_seq_len, pad_id)) %>%
                                # step 11: truncate seq. to maximum length
                                dplyr::slice_head(n = max_seq_len)
  )

  # step 12 convert all to tensors
  # x_feature, we keep xmin, xmax, x_width, x_min_d, x_max_d, x_center_d
  x_features <- torch::torch_stack(purrr::map(encoding_long, ~.x  %>%
                                                dplyr::select(xmin, xmax, x_width, x_min_d, x_max_d, x_center_d) %>%
                                                as.matrix %>%
                                                torch::torch_tensor(dtype = torch::torch_int())))
  # y_feature
  y_features <- torch::torch_stack(purrr::map(encoding_long, ~.x  %>%
                                                dplyr::select(ymin, ymax, y_height, y_min_d, y_max_d, y_center_d) %>%
                                                as.matrix %>%
                                                torch::torch_tensor(dtype = torch::torch_int())))
  # text (used to be input_ids)
  text <- torch::torch_stack(purrr::map(encoding_long, ~.x  %>%
                                          dplyr::select(idx) %>%
                                          as.matrix %>%
                                          torch::torch_tensor(dtype = torch::torch_int())))
  # step 2 + 8 resize and normlize the image for resnet
  image <- torch::torch_stack(purrr::map(seq(nrow(w_h)), ~magick::image_read_pdf(doc, pages = .x) %>%
                                           magick::image_scale(target_geometry) %>%
                                           torchvision::transform_to_tensor() * 255 ))
  # masks
  mask <- torch::torch_stack(purrr::map(encoding_long, ~.x %>%
                                          dplyr::select(mlm_mask) %>%
                                          tidyr::replace_na(list(mlm_mask = TRUE)) %>%
                                          as.matrix %>%
                                          torch::torch_tensor(dtype = torch::torch_bool())))
  # step 13: add tokens for debugging

  # step 14: add extra dim for batch
  encoding_lst <- if (add_batch_dim) {
    list(x_features = x_features, y_features = y_features, text = text, image = image$to(dtype = torch::torch_uint8()), mask = mask)
  } else {
    list(x_features = x_features$squeeze(1), y_features = y_features$squeeze(1), text = text$squeeze(1), image = image$to(dtype = torch::torch_uint8())$squeeze(1), mask = mask$squeeze(1))
  }
  # step 16: void keys to keep, resized_and_aligned_bounding_boxes have been added for the purpose to test if the bounding boxes are drawn correctly or not, it maybe removed
  class(encoding_lst) <- "docformer_tensor"
  attr(encoding_lst, "max_seq_len") <- max_seq_len
  encoding_lst

}
#' Turn DocBanks dataset into docformer torch tensor input feature
#'
#' @param text_path file path or filenames to DocBank_500K_txt
#' @param image_path file path or filenames to the matching DocBank_500K_ori_img
#' @param tokenizer tokenizer function to apply to words extracted from image. Currently,
#'   {hftokenizers}, {tokenizer.bpe} and {sentencepiece} tokenizer are supported.
#' @param add_batch_dim (boolean) add a extra dimension to tensor for batch encoding
#' @param target_geometry image target magik geometry expected by the image model input
#' @param max_seq_len size of the embedding vector in tokens
#' @param batch_size number of images to process
#' @param extras_for_debugging additionnal feature for debugging purposes
#'
#' @return a list of named tensors
#' @export
#'
#' @examples
#' # load a tokenizer with <mask> encoding capability
#' sent_tok <- sentencepiece::sentencepiece_load_model(
#'    system.file(package = "sentencepiece", "models/nl-fr-dekamer.model")
#'    )
#' sent_tok$vocab_size <- sent_tok$vocab_size+1L
#' sent_tok$vocabulary <- rbind(
#'    sent_tok$vocabulary,
#'    data.frame(id = sent_tok$vocab_size, subword = "<mask>")
#'    )
#' # turn pdf into feature
#' text_path <- system.file(package = "docformer", "DocBank_500K_txt")
#' image_path <- system.file(package = "docformer", "DocBank_500K_ori_img")
#' docbanks_tt <- create_features_from_docbank(text_path, image_path, tokenizer = sent_tok)
#'
create_features_from_docbank <- function(text_path,
                                         image_path,
                                         tokenizer,
                                         add_batch_dim = TRUE,
                                         target_geometry = "384x500",
                                         max_seq_len = 512,
                                         batch_size = 1000,
                                         extras_for_debugging = FALSE) {
  # step 0 prepare utilities datasets
  # mask_id <- .mask_id(tokenizer)
  pad_id <- .pad_id(tokenizer)
  txt_col_names <- c("text", "xmin", "ymin", "xmax", "ymax", "font", "class")
  # turn both file_path into file_name vector
  if (fs::is_dir(text_path) && fs::is_dir(image_path)) {
    # list all files in each path
    text_files <- fs::dir_ls(text_path, recurse = TRUE)
    image_path <- text_files %>%
      stringr::str_replace(text_path, image_path) %>%
      stringr::str_replace("\\.txt$", "_ori.jpg")
    text_path <- text_files
  } else if (!fs::is_file(text_path) || !fs::is_file(image_path) ) {
    rlang::abort("text_path is not consistant with image_path. Please review their values")
  }

  # TODO add a coro::loop on length(image_path) %% batch_size to prevent oom
  # step 1 read images and its attributes
  original_image <- purrr::map(image_path, magick::image_read)
  w_h <- purrr::map_dfr(original_image, magick::image_info)
  target_w_h <- stringr::str_split(target_geometry, "x")[[1]] %>%
    as.numeric()

  # TODO: crop and scale each page based on max(xmax)-min(xmin) x max(ymax)-min(ymin)
  # image will be crop to reach alignement
  crop_geometry <- paste0(min(w_h$width),"x",min(w_h$height))
  scale_w <- target_w_h[1] / w_h$width
  scale_h <- target_w_h[2] / w_h$height
  CLS_TOKEN_BOX_long <- c(idx = .cls_id(tokenizer), xmax = target_w_h[1], x_width = target_w_h[1], ymax =target_w_h[2], y_height = target_w_h[2],
                          xmin = 0, ymin = 0, x_min_d = 0, x_max_d = 0, x_center_d = 0, y_min_d = 0, y_max_d = 0, y_center_d = 0)
  SEP_TOKEN_BOX_long <- c(idx = .sep_id(tokenizer), xmax = target_w_h[1], x_width = target_w_h[1], ymax = target_w_h[2], y_height = target_w_h[2],
                          xmin = 0, ymin = 0, x_min_d = 0, x_max_d = 0, x_center_d = 0, y_min_d = 0, y_max_d = 0, y_center_d = 0)

  # step 3 extract text
  # TODO need to transform to lmap with the list(pdftools::pdf_data(doc), scale_w, scale_h) as arguments of an external function
  encoding <-  purrr::pmap(list(as.list(text_path), as.list(scale_w), as.list(scale_h)),
                           ~readr::read_tsv(..1, col_types = "cdddd--cc", col_names = txt_col_names) %>%
                             dplyr::mutate(
                               # step 10 normalize the bbox
                               xmin = trunc(xmin * ..2),
                               ymin = trunc(ymin * ..3),
                               xmax = min(trunc(xmax * ..2),target_w_h[1]),
                               ymax = min(trunc(ymax * ..3),target_w_h[2]),
                               x_center = trunc((xmin + xmax )/2),
                               y_center = trunc((ymin + ymax )/2),
                               # step 11 add relative spatial features
                               x_width = xmax - xmin,
                               y_height = ymax - ymin,
                               x_min_d = dplyr::lead(xmin) - xmin,
                               y_min_d = dplyr::lead(ymin) - ymin,
                               x_max_d = dplyr::lead(xmax) - xmin,
                               y_max_d = dplyr::lead(ymax) - ymin,
                               x_center_d = dplyr::lead(x_center) - x_center,
                               y_center_d = dplyr::lead(y_center) - y_center,
                               # step 4 tokenize words into `idx` and get their bbox
                               idx = .tokenize(tokenizer, text, max_seq_len)) %>%
                             dplyr::select(-x_center, -y_center) %>%
                             tidyr::replace_na(list("", rep(0, 13))))

  encoding_long <- purrr::map(encoding, ~.x  %>%
                                # step 5: apply mask for the sake of pre-training
                                dplyr::mutate(mlm_mask = stats::runif(n = nrow(.x))>0.15) %>%
                                # step 6: unnest tokens
                                tidyr::unnest_longer(col = "idx") %>%
                                # step 7: truncate seq. to maximum length - 2
                                dplyr::slice_head(n = max_seq_len-2) %>%
                                # step 8, 9, 10: prepend sequence with CLS token then append SEP token at last position, then pad to max_seq_len
                                dplyr::bind_rows(CLS_TOKEN_BOX_long, ., SEP_TOKEN_BOX_long, .padding_encode(max_seq_len, pad_id)) %>%
                                # step 11: truncate seq. to maximum length
                                dplyr::slice_head(n = max_seq_len)
  )

  # step 12 convert all to tensors
  # x_feature, we keep xmin, xmax, x_width, x_min_d, x_max_d, x_center_d
  x_features <- torch::torch_stack(purrr::map(encoding_long, ~.x  %>%
                                                dplyr::select(xmin, xmax, x_width, x_min_d, x_max_d, x_center_d) %>%
                                                as.matrix %>%
                                                torch::torch_tensor(dtype = torch::torch_int())))
  # y_feature
  y_features <- torch::torch_stack(purrr::map(encoding_long, ~.x %>%
                                                dplyr::select(ymin, ymax, y_height, y_min_d, y_max_d, y_center_d) %>%
                                                as.matrix %>%
                                                torch::torch_tensor(dtype = torch::torch_int())))
  # text (used to be input_ids)
  text <- torch::torch_stack(purrr::map(encoding_long, ~.x  %>%
                                          dplyr::select(idx) %>%
                                          as.matrix %>%
                                          torch::torch_tensor(dtype = torch::torch_int())))
  # step 8 normlize the image
  image <- torch::torch_stack(purrr::map(seq(nrow(w_h)), ~original_image[[.x]] %>%
                                           magick::image_crop(crop_geometry, gravity = "NorthWest") %>%
                                           magick::image_scale(target_geometry) %>%
                                           torchvision::transform_to_tensor() * 255))
  # masks
  mask <- torch::torch_stack(purrr::map(encoding_long, ~.x %>%
                                          dplyr::select(mlm_mask) %>%
                                          tidyr::replace_na(list(mlm_mask = TRUE)) %>%
                                          as.matrix %>%
                                          torch::torch_tensor(dtype = torch::torch_bool())))
  # step 13: add tokens for debugging

  # step 14: add extra dim for batch
  encoding_lst <- if (add_batch_dim) {
    list(x_features = x_features, y_features = y_features, text = text, image = image$to(dtype = torch::torch_uint8()), mask = mask)
  } else {
    list(x_features = x_features$squeeze(1), y_features = y_features$squeeze(1), text = text$squeeze(1), image = image$to(dtype = torch::torch_uint8())$squeeze(1), mask = mask$squeeze(1))
  }
  # step 16: void keys to keep, resized_and_aligned_bounding_boxes have been added for the purpose to test if the bounding boxes are drawn correctly or not, it maybe removed
  class(encoding_lst) <- "docformer_tensor"
  attr(encoding_lst, "max_seq_len") <- max_seq_len
  encoding_lst

}
#' Save feature tensor to disk
#'
#' @param encoding_lst : the feature tensor list to save
#' @param file : destination file
#'
#' @export
save_featureRDS <- function(encoding_lst, file) {
  # step 15: save to disk
  saveRDS(purrr::map(encoding_lst, ~.x$to(device = "cpu") %>% as.array), file = file)
}

#' Load feature tensor from disk
#'
#' @param file : source file
#'
#' @export
read_featureRDS <- function(file) {
  # step 15: load from disk
  encoding_lst <- readRDS(file = file)
  encoding_lst[1:3] <- encoding_lst[1:3] %>% purrr::map(~torch::torch_tensor(.x,dtype = torch::torch_int()))
  encoding_lst[[4]] <- torch::torch_tensor(encoding_lst[[4]],dtype = torch::torch_uint8())
  encoding_lst[[5]] <- torch::torch_tensor(encoding_lst[[5]],dtype = torch::torch_bool())
  encoding_lst
}

#' @export
mask_for_mm_mlm <- function(encoding_lst, mask_id) {
  # mask tokens idx
  encoding_lst$text <-
    (
      torch::torch_mul(encoding_lst$text, encoding_lst$mask) +
        torch::torch_mul(mask_id, !encoding_lst$mask)
    )$to(torch::torch_int())
  encoding_lst
}

#' @export
mask_for_ltr <- function(encoding_lst) {
  # mask bbox
  batch <- encoding_lst$image$shape[[1]]
  bbox <- torch::torch_cat(list(
    encoding_lst$x_feature[, , 1:1],
    encoding_lst$y_feature[, , 1:1],
    encoding_lst$x_feature[, , 2:2],
    encoding_lst$y_feature[, , 2:2]),
    dim = 3)
  mask_bbox <- purrr::map(
    seq(batch),
    ~ torch::torch_unique_consecutive(bbox[.x:.x, , ]$masked_select(encoding_lst$mask[.x:.x, , ]$logical_not())$view(c(-1, 4)), dim = 1)[[1]][2:N, ]
  )

  encoding_lst$image <- torch::torch_stack(purrr::map(
      seq(batch),
      ~ torchvision::draw_bounding_boxes(
          encoding_lst$image[.x, , , ],
          mask_bbox[[.x]],
          fill = TRUE,
          color = "black"
       )
  ))
    encoding_lst
}
#' @export
mask_for_tdi <- function(encoding_lst) {
  # sample 20 % of the batch
  batch <- encoding_lst$image$shape[[1]]
  is_image_masked <- rbernoulli(batch, p = 0.2)
  randomized_image <- sample(which(!is_image_masked),size = batch, replace = T)
  masked_image_id <- (seq_len(batch) * !is_image_masked) + (randomized_image * is_image_masked)
  # permute switched image with other images from the batch
  encoding_lst$image <- encoding_lst$image[masked_image_id,,,]
  encoding_lst$image_mask <- is_image_masked
  return(encoding_lst)
}

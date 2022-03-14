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
    dplyr::mutate(poor_word =( stringr::str_detect(word, "^\\W+$|\\W{3,}") | confidence <20),
           bb = bbox %>% stringr::str_split(",") %>% purrr::map(as.integer)) %>%
    tidyr::unnest_wider(bb, names_sep="_") %>%
    dplyr::filter(!poor_word) %>%
    dplyr::select(text=word, confidence, xmin=bb_1, ymin=bb_2, xmax=bb_3, ymax=bb_4)
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
  # TODO BUG shall shift-right after max_seq_len slicing
  idx[[1]] <- idx[[1]] %>% purrr::prepend(tokenizer$encode("[CLS]")$ids)
  return(idx)
}
.tokenize.youtokentome <- function(tokenizer, x) {
  idx <- purrr::map(x,~tokenizers.bpe::bpe_encode(tokenizer, .x, type="ids")[[1]])
  # TODO BUG shall shift-right after max_seq_len slicing
  idx[[1]] <- dplyr::first(idx) %>%
    purrr::prepend(tokenizer$vocabulary[tokenizer$vocabulary$subword=="<BOS>",]$id)
  idx[[length(idx)]] <- dplyr::last(idx) %>%
    append(tokenizer$vocabulary[tokenizer$vocabulary$subword=="<EOS>",]$id)
  return(idx)
}
.tokenize.sentencepiece <- function(tokenizer, x) {
  idx <- purrr::map(x,~sentencepiece::sentencepiece_encode(tokenizer, .x, type="ids")[[1]])
  # TODO BUG shall shift-right after max_seq_len slicing
  idx[[1]] <- dplyr::first(idx) %>%
    purrr::prepend(tokenizer$vocabulary[tokenizer$vocabulary$subword=="<s>",]$id)
  idx[[length(idx)]] <- dplyr::last(idx) %>%
    append(tokenizer$vocabulary[tokenizer$vocabulary$subword=="</s>",]$id)
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
  mask_id <- tokenizer$encode("[MASK]")$ids
  if (length(mask_id)==0) {
    rlang::abort("tokenizer do not encode `[MASK]` properly.")
  }
  return(mask_id)
}
.mask_id.youtokentome <- function(tokenizer) {
  mask_id <- tokenizer$vocabulary[tokenizer$vocabulary$subword=="<MASK>",]$id
  if (length(mask_id)==0) {
    rlang::abort("tokenizer do not encode `<MASK>` properly.")
  }
  return(mask_id)
}
.mask_id.sentencepiece <- function(tokenizer) {
  # see https://github.com/google/sentencepiece/blob/master/doc/special_symbols.md for <mask>
  mask_id <- tokenizer$vocabulary[tokenizer$vocabulary$subword=="<mask>",]$id
  if (length(mask_id)==0) {
    rlang::abort("tokenizer do not encode `<mask>` properly.")
  }
  return(mask_id)
}

.empty_encoding <- function(max_seq_len) {
   dplyr::tibble(xmin = rep(0,max_seq_len),
                                  ymin = rep(0,max_seq_len),
                                  xmax = rep(0,max_seq_len),
                                  ymax = rep(0,max_seq_len),
                                  x_width = rep(0,max_seq_len),
                                  y_height = rep(0,max_seq_len),
                                  x_min_d = rep(0,max_seq_len),
                                  y_min_d = rep(0,max_seq_len),
                                  x_max_d = rep(0,max_seq_len),
                                  y_max_d = rep(0,max_seq_len),
                                  x_center_d = rep(0,max_seq_len),
                                  y_center_d = rep(0,max_seq_len),
                                  text = NA_character_,
                                  idx = list(0),
                                  prior=TRUE)


}

#' Turn image into docformer torch tensor input feature
#'
#' @param image file path, url, or raw vector to image (png, tiff, jpeg, etc)
#' @param tokenizer tokenizer function to apply to words extracted from image. Currently,
#'   {hftokenizers}, {tokenizer.bpe} and {sentencepiece} tokenizer are supported.
#' @param add_batch_dim (boolean) add a extra dimension to tensor for batch encoding
#' @param target_geometry image target magik geometry expected by the image model input
#' @param max_seq_len size of the embedding vector in tokens
#' @param apply_mask_for_mlm add mask to the language model
#' @param debugging additionnal feature for debugging purposes
#'
#' @return a list of named tensors
#' @export
#'
#' @examples
#' # load a tokenizer with <mask> encoding capability
#' sent_tok <- sentencepiece::sentencepiece_load_model(system.file(package="sentencepiece", "models/nl-fr-dekamer.model"))
#' sent_tok$vocab_size <- sent_tok$vocab_size+1L
#' sent_tok$vocabulary <- rbind(sent_tok$vocabulary, data.frame(id=sent_tok$vocab_size, subword="<mask>"))
#' # turn pdf into feature
#' image <- system.file(package="docformer", "inst", "2106.11539_1.png")
#' image_tt <- create_features_from_image(image, tokenizer=sent_tok)
#'
create_features_from_image <- function(image,
                            tokenizer,
                            add_batch_dim=TRUE,
                            target_geometry="500x384",
                            max_seq_len=512,
                            apply_mask_for_mlm=FALSE,
                            debugging=FALSE) {

  # step 0 prepare utilities datasets
  mask_id <- .mask_id(tokenizer)
  # step 1 read images and its attributes
  original_image <- magick::image_read(image)
  w_h <- magick::image_info(original_image)
  target_w_h <- stringr::str_split(target_geometry, "x")[[1]] %>%
    as.numeric()
  scale_w <- target_w_h[1] /  w_h$width
  scale_h <- target_w_h[2] / w_h$height
  CLS_TOKEN_BOX <- c(0, 0, w_h$width, w_h$height)   # Can be variable, but as per the paper, they have mentioned that it covers the whole image

  # step 2: resize image
  resized_image <- magick::image_resize(original_image, geometry=target_geometry)

  # step 3 extract text throuhg OCR and normalize bbox to target geometry
  encoding <- apply_ocr(original_image) %>%
    dplyr::mutate(
      # step 10 normalize the bbox
      xmin = round(xmin * scale_w),
      ymin= round(ymin * scale_h),
      xmax = round(xmax * scale_w),
      ymax= round(ymax * scale_h),
      x_center = round((xmin + xmax )/2),
      y_center = round((ymin + ymax )/2),
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
      idx = .tokenize(tokenizer, text)) %>%
    dplyr::select(-confidence, -x_center, -y_center) %>%
    tidyr::replace_na(list("", rep(0, 13)))

  mask_for_mlm <- if (apply_mask_for_mlm) {
    runif(n=nrow(encoding))>0.15
  } else {
    rep(TRUE, nrow(encoding))
  }

  encoding_long <- encoding  %>%
    # step 5.1 apply mask for the sake of pre-training
    dplyr::bind_cols(prior=mask_for_mlm) %>%
    # step 5.2: fill in a max_seq_len matrix
    dplyr::bind_rows(.empty_encoding(max_seq_len)) %>%
    tidyr::unnest_longer(col="idx") %>%
    # step 5.3: truncate seq. to maximum length
    dplyr::slice_head(n=max_seq_len) %>%
    # step 6: (nill here)
    # step 7: apply mask for the sake of pre-training
    dplyr::mutate(idx = ifelse(prior, idx, mask_id))
    # step 8 normlize the image

  # step 12 convert all to tensors
  # x_feature, we keep xmin, xmax, x_width, x_min_d, x_max_d, x_center_d
  x_features <-  encoding_long %>% dplyr::select(xmin, xmax, x_width, x_min_d, x_max_d, x_center_d) %>%
    as.matrix %>% torch::torch_tensor(dtype = torch::torch_double())
  # y_feature
  y_features <- encoding_long %>% dplyr::select(ymin, ymax, y_height, y_min_d, y_max_d, y_center_d) %>%
    as.matrix %>% torch::torch_tensor(dtype = torch::torch_double())
  # text (used to be input_ids)
  text <- encoding_long %>% dplyr::select(idx) %>%
    as.matrix %>% torch::torch_tensor(dtype = torch::torch_double())
  image <- original_image %>% torchvision::transform_resize(size = target_geometry) %>% torchvision::transform_to_tensor()
  # step 13: add tokens for debugging

  # step 14: add extra dim for batch
  encoding_lst <- if (add_batch_dim) {
    list(x_features=x_features$unsqueeze(1), y_features=y_features$unsqueeze(1), text=text$unsqueeze(1), image=image$unsqueeze(1))
  } else {
    list(x_features=x_features, y_features=y_features, text=text, image=image)
  }
  # step 16: void keys to keep, resized_and_aligned_bounding_boxes have been added for the purpose to test if the bounding boxes are drawn correctly or not, it maybe removed
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
#' @param apply_mask_for_mlm add mask to the language model
#' @param extras_for_debugging additionnal feature for debugging purposes
#'
#' @return a list of named tensors
#' @export
#'
#' @examples
#' # load a tokenizer with <mask> encoding capability
#' sent_tok <- sentencepiece::sentencepiece_load_model(system.file(package="sentencepiece", "models/nl-fr-dekamer.model"))
#' sent_tok$vocab_size <- sent_tok$vocab_size+1L
#' sent_tok$vocabulary <- rbind(sent_tok$vocabulary, data.frame(id=sent_tok$vocab_size, subword="<mask>"))
#' # turn pdf into feature
#' doc <- system.file(package="docformer", "inst", "2106.11539_1_2.pdf")
#' doc_tt <- create_features_from_doc(doc, tokenizer=sent_tok)
#'
create_features_from_doc <- function(doc,
                                     tokenizer,
                                     add_batch_dim=TRUE,
                                     target_geometry="384x500",
                                     max_seq_len=512,
                                     apply_mask_for_mlm=FALSE,
                                     extras_for_debugging=FALSE) {
  # step 0 prepare utilities datasets
  mask_id <- .mask_id(tokenizer)

  # step 1 read document and its attributes
  w_h <- pdftools::pdf_pagesize(doc)
  target_w_h <- stringr::str_split(target_geometry, "x")[[1]] %>%
    as.numeric()
  crop_geometry <- paste0(min(w_h$width),"x",min(w_h$height))
  scale_w <- target_w_h[1] / w_h$width
  scale_h <- target_w_h[2] / w_h$height
  CLS_TOKEN_BOX <- c(0, 0, min(w_h$width), min(w_h$height))   # Can be variable, but as per the paper, they have mentioned that it covers the whole image

  # step 3 extract text
  encoding <-  purrr::pmap(list(pdftools::pdf_data(doc), as.list(scale_w), as.list(scale_h)),
                          ~..1 %>% dplyr::mutate(
                            # step 10 normalize the bbox
                            xmin = round( x * ..2),
                            ymin = round( y * ..3),
                            xmax = round((x + width) * ..2),
                            ymax = round((y + height) * ..3),
                            x_center = round((xmin + xmax )/2),
                            y_center = round((ymin + ymax )/2),
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
                            idx = .tokenize(tokenizer, text)) %>%
                            dplyr::select(-x_center, -y_center) %>%
                            tidyr::replace_na(list("", rep(0, 13))))

  mask_for_mlm <- if (apply_mask_for_mlm) {
    purrr::map(encoding, ~runif(n=nrow(.x))>0.15)
  } else {
    purrr::map(encoding, ~rep(TRUE, nrow(.x)))
  }

  encoding <-  purrr::map2(encoding, mask_for_mlm, ~.x %>%
                             # step 5.1 apply mask for the sake of pre-training
                             dplyr::bind_cols(prior=.y) %>%
                             # step 5.2: pad and slice to max_seq_len
                             dplyr::bind_rows(.empty_encoding(max_seq_len))
  )

  encoding_long <- purrr::map(encoding, ~.x  %>%
                                tidyr::unnest_longer(col="idx") %>%
                                # step 5.3: truncate seq. to maximum length
                                dplyr::slice_head(n=max_seq_len) %>%
                                # step 6: (nill here)
                                # step 7: apply mask for the sake of pre-training
                                dplyr::mutate(idx = ifelse(prior, idx, mask_id)))

  # step 12 convert all to tensors
  # x_feature, we keep xmin, xmax, x_width, x_min_d, x_max_d, x_center_d
  x_features <- torch::torch_stack(purrr::map(encoding_long, ~.x  %>% dplyr::select(xmin, xmax, x_width, x_min_d, x_max_d, x_center_d) %>%
                                                as.matrix %>% torch::torch_tensor(dtype = torch::torch_double())))
  # y_feature
  y_features <- torch::torch_stack(purrr::map(encoding_long, ~.x  %>% dplyr::select(ymin, ymax, y_height, y_min_d, y_max_d, y_center_d) %>%
                                                as.matrix %>% torch::torch_tensor(dtype = torch::torch_double())))
  # text (used to be input_ids)
  text <- torch::torch_stack(purrr::map(encoding_long, ~.x  %>% dplyr::select(idx) %>%
                                          as.matrix %>% torch::torch_tensor(dtype = torch::torch_double())))
  # step 2 + 8 resize and normlize the image
  image <- torch::torch_stack(purrr::map(seq(nrow(w_h)), ~magick::image_read_pdf(doc, pages=.x) %>%
                                           # using  Gravity "NorthWestGravity" ensure no shift in x & y
                                           magick::image_crop(crop_geometry, gravity = "NorthWestGravity") %>%
                                           magick::image_scale(target_geometry) %>%
                                           torchvision::transform_to_tensor()))
  # step 13: add tokens for debugging

  # step 14: add extra dim for batch
  encoding_lst <- if (add_batch_dim) {
    list(x_features=x_features, y_features=y_features, text=text, image=image)
  } else {
    list(x_features=x_features$squeeze(1), y_features=y_features$squeeze(1), text=text$squeeze(1), image=image$squeeze(1))
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
#' @param save_to_disk (boolean) shall we save the result onto disk
#' @param path_to_save result path
#' @param apply_mask_for_mlm add mask to the language model
#' @param extras_for_debugging additionnal feature for debugging purposes
#'
#' @return a list of named tensors
#' @export
#'
#' @examples
#' # load a tokenizer with <mask> encoding capability
#' sent_tok <- sentencepiece::sentencepiece_load_model(system.file(package="sentencepiece", "models/nl-fr-dekamer.model"))
#' sent_tok$vocab_size <- sent_tok$vocab_size+1L
#' sent_tok$vocabulary <- rbind(sent_tok$vocabulary, data.frame(id=sent_tok$vocab_size, subword="<mask>"))
#' # turn pdf into feature
#' text_path <- system.file(package="docformer", "DocBank_500K_txt")
#' image_path <- system.file(package="docformer", "DocBank_500K_ori_img")
#' docbanks_tt <- create_features_from_docbank(text_path, image_path, tokenizer=sent_tok)
#'
create_features_from_docbank <- function(text_path,
                                     image_path,
                                     tokenizer,
                                     add_batch_dim=TRUE,
                                     target_geometry="384x500",
                                     max_seq_len=512,
                                     apply_mask_for_mlm=FALSE,
                                     extras_for_debugging=FALSE) {
  # step 0 prepare utilities datasets
  mask_id <- .mask_id(tokenizer)
  txt_col_names <- c("text", "xmin", "ymin", "xmax", "ymax", "font", "class")
  # turn both file_path into file_name vector
  if (is_path(text_path) & is_path(image_path)) {
    # list all files in each path
    text_files <- list.files(text_path, full.names = TRUE, recursive = TRUE)
    image_path <- text_files %>%
      stringr::str_replace(text_path, image_path) %>%
      stringr::str_replace("\\.txt$", "_ori.jpg")
    text_path <- text_files
  } else if (!is_file_list(text_path) | !is_file_list(image_path) ) {
    rlang::abort("text_path is not consistant with image_path. Please review their values")
  }

  # step 1 read images and its attributes
  original_image <- purrr::map(image_path, magick::image_read)
  w_h <- purrr::map_dfr(original_image, magick::image_info)
  target_w_h <- stringr::str_split(target_geometry, "x")[[1]] %>%
    as.numeric()
  crop_geometry <- paste0(min(w_h$width),"x",min(w_h$height))
  scale_w <- target_w_h[1] / w_h$width
  scale_h <- target_w_h[2] / w_h$height
  CLS_TOKEN_BOX <- c(0, 0, min(w_h$width), min(w_h$height))   # Can be variable, but as per the paper, they have mentioned that it covers the whole image

  # step 3 extract text
  # TODO need to transform to lmap with the list(pdftools::pdf_data(doc), scale_w, scale_h) as arguments of an external function
  encoding <-  purrr::pmap(list(as.list(text_path), as.list(scale_w), as.list(scale_h)),
                           ~readr::read_tsv(..1, col_types = "cdddd--cc", col_names = txt_col_names) %>%
                             dplyr::mutate(
                               # step 10 normalize the bbox
                               xmin = round(xmin * ..2),
                               ymin = round(ymin * ..3),
                               xmax = round(xmax * ..2),
                               ymax = round(ymax * ..3),
                               x_center = round((xmin + xmax )/2),
                               y_center = round((ymin + ymax )/2),
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
                               idx = .tokenize(tokenizer, text)) %>%
                             dplyr::select(-x_center, -y_center) %>%
                             tidyr::replace_na(list("", rep(0, 13))))

  mask_for_mlm <- if (apply_mask_for_mlm) {
    purrr::map(encoding, ~runif(n=nrow(.x))>0.15)
  } else {
    purrr::map(encoding, ~rep(TRUE, nrow(.x)))
  }

  encoding <-  purrr::map2(encoding, mask_for_mlm, ~.x %>%
                             # step 5.1 apply mask for the sake of pre-training
                             dplyr::bind_cols(prior=.y) %>%
                             # step 5.2: pad and slice to max_seq_len
                             dplyr::bind_rows(.empty_encoding(max_seq_len))
  )

  encoding_long <- purrr::map(encoding, ~.x  %>%
                                tidyr::unnest_longer(col="idx") %>%
                                # step 5.3: truncate seq. to maximum length
                                dplyr::slice_head(n=max_seq_len) %>%
                                # step 6: (nill here)
                                # step 7: apply mask for the sake of pre-training
                                dplyr::mutate(idx = ifelse(prior, idx, mask_id)))

  # step 12 convert all to tensors
  # x_feature, we keep xmin, xmax, x_width, x_min_d, x_max_d, x_center_d
  x_features <- torch::torch_stack(purrr::map(encoding_long, ~.x  %>% dplyr::select(xmin, xmax, x_width, x_min_d, x_max_d, x_center_d) %>%
                                                as.matrix %>% torch::torch_tensor(dtype = torch::torch_double())))
  # y_feature
  y_features <- torch::torch_stack(purrr::map(encoding_long, ~.x  %>% dplyr::select(ymin, ymax, y_height, y_min_d, y_max_d, y_center_d) %>%
                                                as.matrix %>% torch::torch_tensor(dtype = torch::torch_double())))
  # text (used to be input_ids)
  text <- torch::torch_stack(purrr::map(encoding_long, ~.x  %>% dplyr::select(idx) %>%
                                          as.matrix %>% torch::torch_tensor(dtype = torch::torch_double())))
  # step 8 normlize the image
  image <- torch::torch_stack(purrr::map(seq(nrow(w_h)), ~original_image[[.x]] %>%
                                           # using  Gravity "NorthWestGravity" ensure no shift in x & y
                                           magick::image_crop(crop_geometry, gravity = "NorthWestGravity") %>%
                                           magick::image_scale(target_geometry) %>%
                                           torchvision::transform_to_tensor()))
  # step 13: add tokens for debugging

  # step 14: add extra dim for batch
  encoding_lst <- if (add_batch_dim) {
    list(x_features=x_features, y_features=y_features, text=text, image=image)
  } else {
    list(x_features=x_features$squeeze(1), y_features=y_features$squeeze(1), text=text$squeeze(1), image=image$squeeze(1))
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
#' @return
#' @export
save_featureRDS <- function(encoding_lst, file) {
  # step 15: save to disk
    saveRDS(purrr::map(encoding_lst, ~.x$to(device="cpu") %>% as.array), file = file)
}

#' Load feature tensor from disk
#'
#' @param file : source file
#'
#' @return
#' @export
read_featureRDS <- function(file) {
  # step 15: load from disk
  encoding_lst <- readRDS(file = file)
  encoding_lst[1:3] <- encoding_lst[1:3] %>% purrr::map(~torch::torch_tensor(.x,dtype = torch::torch_double()))
  encoding_lst[[4]] <- torch::torch_tensor(encoding_lst[[4]],dtype = torch::torch_float())
  encoding_lst
}

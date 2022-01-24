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

#' Turn image into docformer torch tensor input feature
#'
#' @param image file path, url, or raw vector to image (png, tiff, jpeg, etc)
#' @param tokenizer tokenizer function to apply to words extracted from image. Currently, only
#'   hftokenizer tokenizer is supported.
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
create_features <- function(image,
                            tokenizer,
                            add_batch_dim=TRUE,
                            target_geometry="224x224",
                            max_seq_len=512,
                            save_to_disk=FALSE,
                            path_to_save=NULL,
                            apply_mask_for_mlm=FALSE,
                            extras_for_debugging=FALSE) {

  # step 1 read images and its attributes
  original_image <- image_read(image)
  w_h <- image_info(original_image)

  # step 2: resize image
  resized_image <- image_resize(original_image, geometry=target_geometry)

  # step 3 extract text throuhg OCR and normalize bbox to 1000x1000
  entries <- apply_ocr(original_image) %>%
    mutate(xmin = round(xmin/w_h$width * 1000),
           ymin= round(ymin/w_h$height * 1000),
           xmax = round(xmax/w_h$width * 1000),
           ymax= round(ymax/w_h$height * 1000)
    )

  # step 4 tokenize words and get their bbox
  ## case hftokenizer
  if (inherits(tokenizer, c("tokenizer", "R6"))) {
    entries <- entries %>%
      mutate(idx = map(entries$word, ~tokenizer$encode(.x)$ids))
  } else {
    rlang::abort(paste0(tokenizer," is not recognized as a supported tokenizer"))
  }

  # token_boxes, unnormalized_token_boxes = get_tokens_with_boxes(unnormalized_word_boxes,
  #                                                               normalized_word_boxes,
  #                                                               PAD_TOKEN_BOX,
  #                                                               encoding.word_ids())
  #
  token_bbox
}

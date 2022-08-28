#' Pipe operator
#'
#' See \code{magrittr::\link[magrittr:pipe]{\%>\%}} for details.
#'
#' @name %>%
#' @rdname pipe
#' @keywords internal
#' @export
#' @importFrom magrittr %>%
#' @usage lhs \%>\% rhs
#'
#' @return Returns `rhs(lhs)`.
NULL

#' @importFrom rlang as_function %||% set_names global_env is_true is_logical
NULL


#' Download and Cache Weights (the torchvision way)
#'
#' @param url the URL of the model to download
#' @param redownload  should the weights be downloaded fresh even if
#'   they're cached? This is not currently exposed to the end user, and exists
#' @param timeout the download timeout
#'
#' @return
#' @export
#'
#' @examples
#' url <- "https://media.githubusercontent.com/media/cregouby/docformer_models/main/inst/tiny-layoutlm.pth"
#' weight <- download_and_cache(url=url)
download_and_cache <- function(url, redownload = FALSE, timeout = 720) {

  cache_path <- rappdirs::user_cache_dir("torch")

  fs::dir_create(cache_path)
  path <- file.path(cache_path, fs::path_file(url))

  if (!file.exists(path) || redownload) {
    withr::with_options(
      list(timeout = timeout),
      utils::download.file(url, path, mode = "wb")
    )
  }
  path
}


#' Download and Cache Weights (the torchtransformer way)
#'
#' Download weights for this model to the torchtransformers cache, or load them
#' if they're already downloaded.
#'
#' @param model_name the name of the model to download or the local file
#' @param redownload Logical; should the weights be downloaded fresh even if
#'   they're cached? This is not currently exposed to the end user, and exists
#' @param timeout Optional timeout in seconds for large file download.
#'
#' @return The parsed weights as a named list.
#' @keywords internal
NULL
# .download_weights <- function(model_name = "microsoft/layoutlm-base-uncased",
#                               redownload = FALSE, timeout = 720) {
#   if (file.exists(model_name)) {
#     return(.process_downloaded_weights(model_name))
#   } else {
#     url <- transformers_config[transformers_config$model_name==model_name,]$url
#     dlr::set_app_cache_dir(appname = "layoutlm", cache_dir = "~/.cache/torch")
#     return(
#       withr::with_options(
#         list(timeout = timeout),
#         dlr::read_or_cache(
#           source_path = url,
#           appname = "layoutlm",
#           process_f = torchtransformers:::.process_downloaded_weights,
#           #read_f = torch::torch_read,
#           write_f = torch::torch_save,
#           write_args = list(use_new_zipfile_serialization=TRUE),
#           force_process = redownload
#         )
#       )
#     )
#   }
# }

#' Process Downloaded Weights
#'
#' @param temp_file The path to the raw downloaded weights.
#'
#' @return The processed weights.
#' @keywords internal
.process_downloaded_weights <- function(temp_file) {
  state_dict <- torch::load_state_dict(temp_file)
  return(state_dict)
}

#' Load Pretrained Weights into a Transformers Model
#'
#' Loads specified pretrained weights into the given BERT model.
#'
#' @param model A transformers-type `nn_module` model.
#' @param model_name Character; which public Transformers model weights to use. Must be compatible
#'   with `model` architecture!. Can be an local file name.
#' @param redownload Logical: Shall we force redownload the model weights ?
#'
#' @return The number of model parameters updated. (This is to enable error
#'   checks; the function is called for side effects.)
#' @keywords internal
.load_weights <- function(model,
                          model_name = "microsoft/layoutlm-base-uncased",
                          redownload = FALSE,
                          timeout = 720) {
  # This will usually just fetch from the cache (torchtransformer way)
  # sd <- .download_weights(model_name = model_name, redownload = redownload)
  # This will usually just fetch from the cache (torchvision way)
  if (!file.exists(model_name)) {
    url <- transformers_config[transformers_config$model_name == model_name, ]$url
    temp_file <- download_and_cache(url = url, redownload = redownload, timeout = timeout)
  } else {
    temp_file <- model_name
  }
  sd <- .process_downloaded_weights(temp_file)

  local_sd <- model$state_dict()
  local_weight_names <- names(local_sd)
  imported_weight_names <- names(sd)
  names_in_common <- intersect(local_weight_names, imported_weight_names)
  if (length(names_in_common) > 0) {
    local_sd[names_in_common] <- sd[names_in_common]
  } else {
    warning("No matching weight names found.")
  }
  model$load_state_dict(local_sd)
}

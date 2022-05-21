#' Configuration for Docformer models
#'
#' @param coordinate_size (int): Output size of each coordinate embedding (default 128)
#' @param shape_size (int): Output size of each position embedding (default 128)
#' @param hidden_dropout_prob (float): Dropout probability in docformer_encoder block (default 0.1)
#' @param attention_dropout_prob (float): Dropout probability in docformer_attention block (default 0.1)
#' @param hidden_size (int): Size of the hidden layer in common with text embedding and positional embedding (default 768)
#' @param image_feature_pool_shape (vector of 3 int): Shqpe of the image feature pooling (default c(7,7,256))
#' @param intermediate_ff_size_factor (int): Intermediate feed-forward layer expension factor (default 3)
#' @param max_2d_position_embeddings (int): Max size of vector hosting the 2D embedding (default 1024)
#' @param max_position_embeddings (int): Max sequence length for 1D embedding (default 512)
#' @param max_relative_positions (int): Max number of position to look at in multimodal attention layer (default 8)
#' @param num_attention_heads (int): Number of attention heads in the encoder (default 12)
#' @param num_hidden_layers (int): Number of attention layers in the encoder
#' @param pad_token_id (int): Id of the padding token
#' @param vocab_size (int): Length of the vocabulary
#' @param type_vocab_size (int): Length of the type vocabulary
#' @param layer_norm_eps (float): Epsilon value used in normalisation layer (default 1e-12)
#' @param batch_size (int): Size of the batch.
#' @param loss (character or function) Loss function for training (default to mse
#'   for regression and cross entropy for classification)
#' @param epochs (int) Number of training epochs.
#' @param pretraining_ratio (float): Ratio of features to mask for reconstruction during
#'   pretraining.  Ranges from 0 to 1 (default=0.5)
#' @param verbose (bool): Whether to print progress and loss values during
#'   training.
#' @param device The device to use for training. "cpu" or "cuda". The default ("auto")
#'   uses  to "cuda" if it's available, otherwise uses "cpu".
#' @param pretrained_model_name (character) : one of the supported model name in `transformers_config` to derive config from.
#'
#' @return a named list will all needed hyperparameters of the Docformer implementation.
#' @export
#'
#' @examples
#' config <- docformer_config(
#'   num_attention_heads=6L,
#'   num_hidden_layers=6L,
#'   batch_size=27,
#'   epoch =5,
#'   verbose=TRUE
#'   )
#' config <- docformer_config(
#'   pretrained_model_name="hf-internal-testing/tiny-layoutlm",
#'   batch_size=27,
#'   epoch =5
#'   )
#'
docformer_config <- function(pretrained_model_name=NA_character_,
                             coordinate_size = 128L,
                             shape_size = 128L,
                             hidden_dropout_prob = 0.1,
                             attention_dropout_prob = 0.1,
                             hidden_size = 768L,
                             image_feature_pool_shape = c(7, 7, 256),
                             intermediate_ff_size_factor = 4L,  # could be turned to 3L
                             max_2d_position_embeddings = 1024L,
                             max_position_embeddings = 512L,
                             max_relative_positions = 8L,
                             num_attention_heads = 12L,
                             num_hidden_layers = 12L,
                             pad_token_id = 1L,
                             vocab_size = 30522L,
                             type_vocab_size = 2L,
                             layer_norm_eps = 1e-12,
                             batch_size = 9L,
                             loss = "auto",
                             epochs = 5,
                             pretraining_ratio = 0.5,
                             verbose = FALSE,
                             device = "auto"
) {
  # override config parameters from pretrained model if any
  if (!is.na(pretrained_model_name)) {
    if (pretrained_model_name %in% transformers_config$model_name) {
      transformer_c <- transformers_config %>% dplyr::filter(model_name == pretrained_model_name)
      hidden_size <- transformer_c$hidden_size
      shape_size <- hidden_size %/% 6
      coordinate_size <- (hidden_size - 4 * shape_size)/2
      intermediate_ff_size_factor <-transformer_c$intermediate_ff_size_factor
      max_2d_position_embeddings <- transformer_c$max_2d_position_embeddings
      max_position_embeddings <- transformer_c$max_position_embeddings
      num_attention_heads <- transformer_c$n_head
      num_hidden_layers <- transformer_c$n_layer
      vocab_size <- transformer_c$vocab_size
    } else {
      rlang::warn("Provided model name cannot be found in `transformers_config`. using default config values")
    }
  } else {
    pretrained_model_name <- "microsoft/layoutlm-base-uncased"
  }
  # consistency check
  if (hidden_size %% num_attention_heads !=0) {
    rlang::abort(message="Error: `hidden_size` is not multiple of `num_attention_heads` which prevent initialization of the multimodal_attention_layer")
  }

  if (2 * coordinate_size + 4 * shape_size != hidden_size) {
    rlang::abort(message="Error: `coordinate_size` x 2 +  `shape_size` x 4 do not equal `hidden_size`")
  }

  # resolve device
  if (device == "auto") {
    if (torch::cuda_is_available()){
      device <- "cuda"
    } else {
      device <- "cpu"
    }
  } else {
    device <- device
  }
  list(
    coordinate_size = coordinate_size,
    hidden_dropout_prob = hidden_dropout_prob,
    attention_dropout_prob = attention_dropout_prob,
    hidden_size = hidden_size,
    image_feature_pool_shape = image_feature_pool_shape,
    intermediate_ff_size_factor = intermediate_ff_size_factor,
    max_2d_position_embeddings = max_2d_position_embeddings,
    max_position_embeddings = max_position_embeddings,
    max_relative_positions = max_relative_positions,
    num_attention_heads = num_attention_heads,
    num_hidden_layers = num_hidden_layers,
    pad_token_id = pad_token_id,
    shape_size = shape_size,
    vocab_size = vocab_size,
    type_vocab_size = type_vocab_size,
    layer_norm_eps = layer_norm_eps,
    batch_size = batch_size,
    pretraining_ratio = pretraining_ratio,
    verbose = verbose,
    device = device,
    is_decoder = FALSE,
    intermediate_size = intermediate_ff_size_factor * hidden_size,
    hidden_act = "gelu",
    num_labels = 1L,
    pretrained_model_name = pretrained_model_name
  )


}

#' Docformer model
#'
#' Fits the [DocFormer: End-to-End Transformer for Document Understanding](https://arxiv.org/abs/2106.11539) model
#'
#' @param x Depending on the context:
#'
#'   * A __image__ filename.
#'   * A __document__ filename.
#'   * A __folder__ containing either images or documents.
#'
#'  The model currently support for __image__ any image type that `{magick}` package can read.
#'  The model currently support for __document__ any pdf type that `{pdftool}` package can read.
#'
#' @param y A __data frame__
#' @param docformer_model A previously fitted DocFormer model object to continue the fitting on.
#'  if `NULL` (the default) a brand new model is initialized.
#' @param config A set of hyperparameters created using the `docformer_config` function.
#'  If no argument is supplied, this will use the default values in [docformer_config()].
#' @param from_epoch When a `docformer_model` is provided, restore the network weights from a specific epoch.
#'  Default is last available checkpoint for restored model, or last epoch for in-memory model.
#' @param ... Model hyperparameters.
#' Any hyperparameters set here will update those set by the config argument.
#' See [docformer_config()] for a list of all possible hyperparameters.
#'
#' @section Fitting a pre-trained model:
#'
#' When providing a parent `docformer_model` parameter, the model fitting resumes from that model weights
#' at the following epoch:
#'    * last fitted epoch for a model already in torch context
#'    * Last model checkpoint epoch for a model loaded from file
#'    * the epoch related to a checkpoint matching or preceding the `from_epoch` value if provided
#' The model fitting metrics append on top of the parent metrics in the returned TabNet model.
#'
#' @examples
#' docformer_model <- docformer_fit(x)
#' @return A DocFormer model object of class `docformer_fit` It can be used for serialization, predictions, or further fitting.
#'
#' @export
docformer_fit <- function(x, ...) {
  UseMethod("docformer_fit")
}
#' @export
#' @rdname docformer_fit
docformer_fit.default <- function(x, ...) {
  stop(
    "`docformer_fit()` is not defined for a '", class(x)[1], "'.",
    call. = FALSE
  )
}

#' @export
#' @rdname docformer_fit
docformer_fit.docformer_tensor <- function(x, config = docformer_config(), ...) {
  # assemble config and ...
  default_config <- docformer_config()
  new_config <- do.call(docformer_config, list(...))
  new_config <- new_config[
    mapply(
      function(x, y) ifelse(is.null(x), !is.null(y), x != y),
      default_config,
      new_config)
  ]
  config <- utils::modifyList(config, as.list(new_config))
  # luz training
  docformer(config) %>%
    luz::setup(
      loss = torch::nn_mse_loss(),
      optimizer = torch::optim_adam
    ) %>%
    luz::set_hparams(config = config) %>%
    luz::fit(x, epochs = config$epoch)
}
#
#' @importFrom stats predict
#' @export
predict.docformer_fit <- function(object, new_data, type = NULL, ..., epoch = NULL) {

}

#' @export
print.docformer_fit <- function(x, ...) {
  if (check_net_is_empty_ptr(x)) {
    print(reload_model(x$serialized_net))
  } else {
    print(x$fit$network)
  }
  invisible(x)
}
#' @export
print.docformer_pretrain <- print.docformer_fit

# docformer_initialize <- function(x, y, config = tabnet_config()){
#   has_valid <- config$valid_split > 0
#
#   if (config$device == "auto") {
#     if (torch::cuda_is_available()){
#       device <- "cuda"
#     } else {
#       device <- "cpu"
#     }
#   } else {
#     device <- config$device
#   }
#
#   # simplify y into vector
#   if (!is.atomic(y)) {
#     # currently not supporting multilabel
#     y <- y[[1]]
#   }
#
#   if (has_valid) {
#     n <- nrow(x)
#     valid_idx <- sample.int(n, n*config$valid_split)
#     valid_x <- x[valid_idx, ]
#     valid_y <- y[valid_idx]
#     train_y <- y[-valid_idx]
#     valid_ds <-   torch::dataset(
#       initialize = function() {},
#       .getbatch = function(batch) {resolve_data(valid_x[batch,], valid_y[batch], device=device)},
#       .length = function() {nrow(valid_x)}
#     )()
#     x <- x[-valid_idx, ]
#     y <- train_y
#   }
#
#   # training dataset
#   train_ds <-   torch::dataset(
#     initialize = function() {},
#     .getbatch = function(batch) {resolve_data(x[batch,], y[batch], device=device)},
#     .length = function() {nrow(x)}
#   )()
#   # we can get training_set parameters from the 2 first samples
#   train <- train_ds$.getbatch(batch = c(1:2))
#
#   # resolve loss
#   config$loss_fn <- resolve_loss(config$loss, train$y$dtype)
#
#   # create network
#   network <- docformer(config)
#
#   # main loop
#   metrics <- list()
#   checkpoints <- list()
#
#   list(
#     network = network,
#     metrics = metrics,
#     config = config,
#     checkpoints = checkpoints
#   )
# }



# docformer_train_supervised <- function(data_loader, model, criterion, optimizer, epoch, device, scheduler=None){
#
# }

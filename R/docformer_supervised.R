#' Configuration fro Docformer models
#'
#' @param coordinate_size (int): Output size  of each coordinate embedding (default 96)
#' @param shape_size (int): Output size of each position embedding (default 96)
#' @param hidden_dropout_prob (float): Dropout probability in docformer_encoder block (default 0.1)
#' @param hidden_size (int): Size of the hidden layer in common with text embedding and positional embedding (default 768)
#' @param image_feature_pool_shape (vector of 3 int): Shqpe of the image feature pooling (default c(7,7,256))
#' @param intermediate_ff_size_factor (int): Intermediate feed-forward layer expension factor (default 3)
#' @param max_2d_position_embeddings (int): Max size of vector hosting the 2D embedding (default 1024)
#' @param max_position_embeddings (int): Max sequence length for 1D embedding (default 512)
#' @param max_relative_positions (int): Max number of position to look at in multimodal attention layer (default 8)
#' @param num_attention_heads (int): Number of attention heads (default 12)
#' @param num_hidden_layers (int): Number of hidden layers in the encoder
#' @param pad_token_id (int): Id of the padding token
#' @param vocab_size (int): Length of the vocabulary
#' @param layer_norm_eps (float): Epsilon value used in normalisation layer (default 1e-12)
#' @param batch_size (int): Size of the batch (default 7)
#' @param pretraining_ratio (float): Ratio of features to mask for reconstruction during
#'   pretraining.  Ranges from 0 to 1 (default=0.5)
#' @param verbose (bool): Whether to print progress and loss values during
#'   training.
#' @param device The device to use for training. "cpu" or "cuda". The default ("auto")
#'   uses  to "cuda" if it's available, otherwise uses "cpu".
#'
#' @return a named list will all needed hyperparameters of the Docformer implementation.
#' @export
#'
#' @examples
docformer_config <- function(    coordinate_size = 96L,
                                 shape_size = 96L,
                                 hidden_dropout_prob = 0.1,
                                 hidden_size = 768L,
                                 image_feature_pool_shape = c(7, 7, 256),
                                 intermediate_ff_size_factor = 3L,  # default ought to be 4
                                 max_2d_position_embeddings = 1024L,
                                 max_position_embeddings = 512L,
                                 max_relative_positions = 8L,
                                 num_attention_heads = 12L,
                                 num_hidden_layers = 12L,
                                 pad_token_id = 1L,
                                 vocab_size = 30522L,
                                 layer_norm_eps = 1e-12,
                                 batch_size = 9L,
                                 pretraining_ratio = 0.5,
                                 verbose = FALSE,
                                 device = "auto"
) {
  list(
    coordinate_size = coordinate_size,
    hidden_dropout_prob = hidden_dropout_prob,
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
    layer_norm_eps = layer_norm_eps,
    batch_size = batch_size,
    pretraining_ratio = pretraining_ratio,
    verbose = verbose,
    device = device
  )

}

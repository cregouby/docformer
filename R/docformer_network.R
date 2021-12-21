positional_encoding <- torch::nn_module(
  "positional_encoding",
  initialize = function(d_model, dropout=0.1, max_len=5000){
    self$dropout <- torch::nn_dropout(p=dropout)
    self$max_len <- max_len
    self$d_model <- d_model
    position <- torch::torch_arange(start=1, end = max_len)$unsqueeze(2)
    div_term <- torch::torch_exp(torch::torch_arange(1, d_model, 2) * (-log(1e5)/d_model))
    pe <- torch::torch_zeros(1, max_len, d_model)
    pe[1,,1:N:2] <- torch::torch_sin(position * div_term)
    pe[1,,2:N:2] <- torch::torch_cos(position * div_term)
    self$pe <-  pe
  },
  forward = function() {
    x <- self$pe[1, 1:self$max_len]
    self$dropout(x)$unsqueeze(1)
  }
)
resnet_feature_extractor <- torch::nn_module(
  "resnet_feature_extractor",
  initialize = function(){
    self$image_size <- c(3,224,224)
    # use ResNet model for visual features embedding (remove classificaion head)
    resnet50 <- torchvision::model_resnet50(pretrain=TRUE)
    modules <- torch::nn_module_list(resnet50$children)
    self$resnet50 <- torch::nn_sequential(modules[1:(length(modules)-2)])
    # Applying convolution and linear layer
    self$conv1 <- torch::nn_conv2d(2048,768,1)
    self$relu1 <- torch::torch_relu()
    self$linear1 <- torch::nn_linear(49,512)
  },
  forward = function(x) {
    x %>%
      self$resnet50 %>%
      self$conv1 %>%
      self$relu1 %>%
      torch::torch_reshape(c(x$size(1:2), -1)) %>% # "b e w h -> b e (w.h)" batch, embedding, w, h
      self$linear1 %>%
      torch::torch_flip(c(2,3)) # "b e s -> b s e", batch, embedding, sequence
  }
)
docformer_embeddings <- torch::nn_module(
  "docformer_embeddings",
  initialize = function(config){
    self$config <- config
    max_2d_p_emb <- config$max_2d_position_embeddings

    self$word_embedding <- torch::nn_embedding(config$vocab_size, config$hidden_size, padding_idx = config$pad_token_id)
    self$position_embedding_v <- positional_encoding(d_model=config$hidden_size, dropout=0.1, max_len=config$max_position_embeddings)

    self$x_v <- torch::nn_embedding(max_2d_p_emb, config$coordinate_size)
    self$x_topleft_position_embeddings_v <- torch::nn_embedding(max_2d_p_emb, config$coordinate_size)
    self$x_bottomright_position_embeddings_v <- torch::nn_embedding(max_2d_p_emb, config$coordinate_size)
    self$w_position_embeddings_v <- torch::nn_embedding(max_2d_p_emb, config$shape_size)
    self$x_topleft_distance_to_prev_embeddings_v <- torch::nn_embedding(max_2d_p_emb, config$shape_size)
    self$x_bottomleft_distance_to_prev_embeddings_v <- torch::nn_embedding(max_2d_p_emb, config$shape_size)
    self$x_topright_distance_to_prev_embeddings_v <- torch::nn_embedding(max_2d_p_emb, config$shape_size)
    self$x_bottomright_distance_to_prev_embeddings_v <- torch::nn_embedding(max_2d_p_emb, config$shape_size)
    self$x_centroid_distance_to_prev_embeddings_v <- torch::nn_embedding(max_2d_p_emb, config$shape_size)

    self$y_topleft_position_embeddings_v <- torch::nn_embedding(max_2d_p_emb, config$coordinate_size)
    self$y_bottomright_position_embeddings_v <- torch::nn_embedding(max_2d_p_emb, config$coordinate_size)
    self$h_position_embeddings_v <- torch::nn_embedding(max_2d_p_emb, config$shape_size)
    self$y_topleft_distance_to_prev_embeddings_v <- torch::nn_embedding(max_2d_p_emb, config$shape_size)
    self$y_bottomleft_distance_to_prev_embeddings_v <- torch::nn_embedding(max_2d_p_emb, config$shape_size)
    self$y_topright_distance_to_prev_embeddings_v <- torch::nn_embedding(max_2d_p_emb, config$shape_size)
    self$y_bottomright_distance_to_prev_embeddings_v <- torch::nn_embedding(max_2d_p_emb, config$shape_size)
    self$y_centroid_distance_to_prev_embeddings_v <- torch::nn_embedding(max_2d_p_emb, config$shape_size)

    self$position_embedding_t <- positional_encoding(d_model=config$hidden_size, dropout=0.1, max_len=config$max_position_embeddings)

    self$x_topleft_position_embeddings_t <- torch::nn_embedding(max_2d_p_emb, config$coordinate_size)
    self$x_bottomright_position_embeddings_t <- torch::nn_embedding(max_2d_p_emb, config$coordinate_size)
    self$w_position_embeddings_t <- torch::nn_embedding(max_2d_p_emb, config$shape_size)
    self$x_topleft_distance_to_prev_embeddings_t <- torch::nn_embedding(max_2d_p_emb, config$shape_size)
    self$x_bottomleft_distance_to_prev_embeddings_t <- torch::nn_embedding(max_2d_p_emb, config$shape_size)
    self$x_topright_distance_to_prev_embeddings_t <- torch::nn_embedding(max_2d_p_emb, config$shape_size)
    self$x_bottomright_distance_to_prev_embeddings_t <- torch::nn_embedding(max_2d_p_emb, config$shape_size)
    self$x_centroid_distance_to_prev_embeddings_t <- torch::nn_embedding(max_2d_p_emb, config$shape_size)

    self$y_topleft_position_embeddings_t <- torch::nn_embedding(max_2d_p_emb, config$coordinate_size)
    self$y_bottomright_position_embeddings_t <- torch::nn_embedding(max_2d_p_emb, config$coordinate_size)
    self$h_position_embeddings_t <- torch::nn_embedding(max_2d_p_emb, config$shape_size)
    self$y_topleft_distance_to_prev_embeddings_t <- torch::nn_embedding(max_2d_p_emb, config$shape_size)
    self$y_bottomleft_distance_to_prev_embeddings_t <- torch::nn_embedding(max_2d_p_emb, config$shape_size)
    self$y_topright_distance_to_prev_embeddings_t <- torch::nn_embedding(max_2d_p_emb, config$shape_size)
    self$y_bottomright_distance_to_prev_embeddings_t <- torch::nn_embedding(max_2d_p_emb, config$shape_size)
    self$y_centroid_distance_to_prev_embeddings_t <- torch::nn_embedding(max_2d_p_emb, config$shape_size)

    self$layer_norm <- torch::nn_layer_norm(config$hidden_size, eps=config$layer_norm_eps)
    self$dropout <- torch::nn_dropout(config$hidden_dropout_prob)

    self$x_embedding_v <- list(
      self$x_topleft_position_embeddings_v,
      self$x_bottomright_position_embeddings_v,
      self$w_position_embeddings_v,
      self$x_topleft_distance_to_prev_embeddings_v,
      self$x_bottomleft_distance_to_prev_embeddings_v,
      self$x_topright_distance_to_prev_embeddings_v,
      self$x_bottomright_distance_to_prev_embeddings_v,
      self$x_centroid_distance_to_prev_embeddings_v
    )
    self$y_embedding_v <- list(
      self$y_topleft_position_embeddings_v,
      self$y_bottomright_position_embeddings_v,
      self$h_position_embeddings_v,
      self$y_topleft_distance_to_prev_embeddings_v,
      self$y_bottomleft_distance_to_prev_embeddings_v,
      self$y_topright_distance_to_prev_embeddings_v,
      self$y_bottomright_distance_to_prev_embeddings_v,
      self$y_centroid_distance_to_prev_embeddings_v
    )
    self$x_embedding_t <- list(
      self$x_topleft_position_embeddings_t,
      self$x_bottomright_position_embeddings_t,
      self$w_position_embeddings_t,
      self$x_topleft_distance_to_prev_embeddings_t,
      self$x_bottomleft_distance_to_prev_embeddings_t,
      self$x_topright_distance_to_prev_embeddings_t,
      self$x_bottomright_distance_to_prev_embeddings_t,
      self$x_centroid_distance_to_prev_embeddings_t
    )
    self$y_embedding_t <- list(
      self$y_topleft_position_embeddings_t,
      self$y_bottomright_position_embeddings_t,
      self$h_position_embeddings_t,
      self$y_topleft_distance_to_prev_embeddings_t,
      self$y_bottomleft_distance_to_prev_embeddings_t,
      self$y_topright_distance_to_prev_embeddings_t,
      self$y_bottomright_distance_to_prev_embeddings_t,
      self$y_centroid_distance_to_prev_embeddings_t
    )

  },
  forward = function(x_feature, y_feature) {
    # Arguments:
    #   x_features of shape (batch_size, seq_len, 8)
    # y_features of shape (batch_size, seq_len, 8)
    #
    # Outputs:
    #
    #   (V-bar-s, T-bar-s) of shape (batch_size, 512,768),(batch_size, 512,768)
    #
    # What are the 8 features:
    #
    # 1 -> top left x/y
    # 2 -> bottom right x/y
    # 3 -> width/height
    # 4 -> diff top left x/y
    # 5 -> diff bottom left x/y
    # 6 -> diff top right x/y
    # 7 -> diff bottom right x/y
    # 8 -> centroids diff x/y

    batch <- x_feature.shape(1)
    seq_len  <-  x_feature.shape(2)
    num_feat  <-  x_feature.shape(3) # 8
    hidden_size  <-  self$config$hidden_size
    sub_dim  <-  hidden_size %/% num_feat

    x_calculated_embedding_v  <-  torch::torch_zeros(batch, seq_len, hidden_size, device = x_feature$device)
    y_calculated_embedding_v  <-  torch::torch_zeros(batch, seq_len, hidden_size, device = x_feature$device)
    x_calculated_embedding_t  <-  torch::torch_zeros(batch, seq_len, hidden_size, device = x_feature$device)
    y_calculated_embedding_t  <-  torch::torch_zeros(batch, seq_len, hidden_size, device = x_feature$device)

    for (i in seq(num_feat)) {
      x_calculated_embedding_v[.., i * sub_dim] <-  self$x_embedding_v[[i]](x_feature[.., i])
      y_calculated_embedding_v[.., i * sub_dim] <-  self$y_embedding_v[[i]](y_feature[.., i])
      x_calculated_embedding_t[.., i * sub_dim] <-  self$x_embedding_t[[i]](x_feature[.., i])
      y_calculated_embedding_t[.., i * sub_dim] <-  self$y_embedding_t[[i]](y_feature[.., i])
    }
    v_bar_s <-  x_calculated_embedding_v + y_calculated_embedding_v + self.position_embeddings_v()
    t_bar_s <-  x_calculated_embedding_t + y_calculated_embedding_t + self.position_embeddings_t()

    return(v_bar_s,t_bar_s)

  }
)
pre_norm <- torch::nn_module(
  "pre_norm",
  initialize = function(dim, fn){
    self$norm <- torch::nn_layer_norm(dim)
    self$fn <- fn
  },
  forward = function(x,...) {
    return(self$fn(self$norm(x)), ...)
  }
)
pre_norm_attention <- torch::nn_module(
  "pre_norm_attention",
  initialize = function(dim, fn){
    self$norm_t_bar <- torch::nn_layer_norm(dim)
    self$norm_v_bar <- torch::nn_layer_norm(dim)
    self$norm_t_bar_s <- torch::nn_layer_norm(dim)
    self$norm_v_bar_s <- torch::nn_layer_norm(dim)
    self$fn <- fn
  },
  forward = function(t_bar, v_bar, t_bar_s, v_bar_s, ...) {
    return(self$fn(
      self$norm_t_bar(t_bar),
      self$norm_v_bar(v_bar),
      self$norm_t_bar_s(t_bar_s),
      self$norm_v_bar_s(v_bar_s),
    ), ...)

  }
)

feed_forward <- torch::nn_module(
  "feed_forward",
  initialize = function(dim, hidden_dim, dropout=0){
    self$net <- torch::nn_sequential(
      torch::nn_linear(dim, hidden_dim),
      torch::nn_gelu(),
      torch::nn_dropout(dropout),
      torch::nn_linear(hidden_dim, dim),
      torch::nn_dropout(dropout)
    )
  },
  forward = function(x) {
    self$net(x)
  }
)

relative_position <- torch::nn_module(
  "relative_position",
  initialize = function(num_units, max_relative_position, max_seq_length){
    self$num_units <- num_units
    self$max_relative_position <- max_relative_position
    self$embeddings_table <- torch::nn_parameter(torch::torch_zeros(c(max_relative_position *2 +1, num_units)))
    torch::nn_init_xavier_uniform_(self$embeddings_table)

    self$max_length <- max_seq_length
    range_vec_q  <-  torch::torch_arange(1, max_seq_length)
    range_vec_k  <-  torch::torch_arange(1, max_seq_length)
    distance_mat  <-  range_vec_k[NULL, ..] - range_vec_q[.., NULL]
    distance_mat_clipped  <-  torch::torch_clamp(distance_mat, -self$max_relative_position, self$max_relative_position)
    final_mat  <-  distance_mat_clipped + self$max_relative_position
    self$final_mat  <-  final_mat$to(dtype=torch_long())
  },
  forward = function(length_q, length_k) {
    embeddings  <-  self$embeddings_table[self$final_mat[1:length_q, 1:length_k]]
  }
)

multimodal_attention_layer <- torch::nn_module(
  "multimodal_attention_layer",
  initialize = function(){

  },
  forward = function() {

  }
)
docformer_encoder <- torch::nn_module(
  "docformer_encoder",
  initialize = function(){

  },
  forward = function() {

  }
)

language_feature_extractor <- torch::nn_module(
  "language_feature_extractor",
  initialize = function(){

  },
  forward = function() {

  }
)
extract_features <- torch::nn_module(
  "extract_features",
  initialize = function(){

  },
  forward = function() {

  }
)

docformer_for_classification <- torch::nn_module(
  "docformer_for_classification",
  initialize = function(){

  },
  forward = function() {

  }
)

docformer_for_ir <- torch::nn_module(
  "docformer_for_ir",
  initialize = function(){

  },
  forward = function() {

  }
)

docformer <- torch::nn_module(
  "docformer",
  initialize = function(){

  },
  forward = function() {

  }
)

shallow_decoder <- torch::nn_module(
  "shallow_decoder",
  initialize = function(){

  },
  forward = function() {

  }
)

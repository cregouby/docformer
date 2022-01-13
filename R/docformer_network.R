positional_encoding <- torch::nn_module(
  "positional_encoding",
  initialize = function(d_model, dropout=0.1, max_len=5000){
    self$dropout <- torch::nn_dropout(p=dropout)
    self$max_len <- max_len
    self$d_model <- d_model
    position <- torch::torch_arange(start=1, end = max_len)$unsqueeze(2)
    div_term <- torch::torch_exp(torch::torch_arange(1, d_model, 2) * (-log(1e5)/d_model))
    pe <- torch::torch_zeros(1, max_len, d_model, device = config$device)
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
      torch::torch_movedim(c(2,3)) # "b e s -> b s e", batch, embedding, sequence
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

    # unused
    # self$layer_norm <- torch::nn_layer_norm(config$hidden_size, eps=config$layer_norm_eps)
    # self$dropout <- torch::nn_dropout(config$hidden_dropout_prob)

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

    x_calculated_embedding_v  <-  torch::torch_zeros(batch, seq_len, hidden_size, device = config$device)
    y_calculated_embedding_v  <-  torch::torch_zeros(batch, seq_len, hidden_size, device = config$device)
    x_calculated_embedding_t  <-  torch::torch_zeros(batch, seq_len, hidden_size, device = config$device)
    y_calculated_embedding_t  <-  torch::torch_zeros(batch, seq_len, hidden_size, device = config$device)

    for (i in seq(num_feat)) {
      x_calculated_embedding_v[.., i * sub_dim] <-  self$x_embedding_v[[i]](x_feature[.., i])
      y_calculated_embedding_v[.., i * sub_dim] <-  self$y_embedding_v[[i]](y_feature[.., i])
      x_calculated_embedding_t[.., i * sub_dim] <-  self$x_embedding_t[[i]](x_feature[.., i])
      y_calculated_embedding_t[.., i * sub_dim] <-  self$y_embedding_t[[i]](y_feature[.., i])
    }
    v_bar_s <-  x_calculated_embedding_v + y_calculated_embedding_v + self.position_embeddings_v()
    t_bar_s <-  x_calculated_embedding_t + y_calculated_embedding_t + self.position_embeddings_t()

    return(list(v_bar_s,t_bar_s))

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
    self$embeddings_table <- torch::nn_parameter(torch::torch_zeros(c(max_relative_position *2 +1, num_units), device=config$device))
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
  initialize = function(embed_dim, n_heads, max_relative_position, max_seq_length, dropout){
    stopifnot("Error: `embed_dim` is not multiple of `n_head` which prevent initialization of the multimodal_attention_layer" = embed_dim %% n_heads ==0)

    self$embed_dim <- embed_dim
    self$n_heads <- n_heads
    self$head_dim <- head_dim %/% n_heads

    self$relative_positions_text  <-  relative_position(self$head_dim, max_relative_position, max_seq_length)
    self$relative_positions_img <- relative_position(self$head_dim, max_relative_position, max_seq_length)

    # text qkv embeddings
    self$fc_k_text <- torch::nn_linear(embed_dim, embed_dim)
    self$fc_q_text <- torch::nn_linear(embed_dim, embed_dim)
    self$fc_v_text <- torch::nn_linear(embed_dim, embed_dim)

    # image qkv embeddings
    self$fc_k_img <- torch::nn_linear(embed_dim, embed_dim)
    self$fc_q_img <- torch::nn_linear(embed_dim, embed_dim)
    self$fc_v_img <- torch::nn_linear(embed_dim, embed_dim)

    # spatial qk embeddings (shared for visual and text)
    self$fc_k_spatial <- torch::nn_linear(embed_dim, embed_dim)
    self$fc_q_spatial <- torch::nn_linear(embed_dim, embed_dim)

    self$softmax_dropout <- torch::nn_sequential(
      torch::torch_softmax(dim=-1),
      torch::nn_dropout(dropout)
    )

    self$to_out <- torch::nn_sequential(
      torch::nn_linear(embed_dim, embed_dim),
      torch::nn_dropout(dropout)
    )
    self$scale <- torch::torch_sqrt(torch::torch_tensor(embed_dim, device = config$device))


  },
  forward = function(text_feat, img_feat, text_spatial_feat, img_spatial_feat) {
    seq_length <- text_feat$shape[2]

    # self attention of text
    # b -> batch, t -> time steps (l -> length has same meaning), head -> # of heads, k -> head dim.
    # 'b t (head k) -> head b t k'
    key_text_nh_bthk <- self$fc_k_text(text_feat)$unsqueeze(4)
    dim <- key_text_nh_bthk$shape
    key_text_nh <- key_text_nh_bthk$reshape(c(dim[1], dim[2], self$n_heads, -1))$permute(c(3,1,2,4))
    # 'b l (head k) -> head b l k'
    query_text_nh_blhk <- self$fc_q_text(text_feat)$unsqueeze(4)
    dim <- query_text_nh_blhk$shape
    query_text_nh <- query_text_nh_blhk$reshape(c(dim[1], dim[2], self$n_heads, -1))$permute(c(3,1,2,4))
    # 'b t (head k) -> head b t k'
    value_text_nh_bthk <- self$fc_v_text(text_feat)$unsqueeze(4)
    dim <- value_text_nh_bthk$shape
    value_text_nh <- value_text_nh_bthk$reshape(c(dim[1], dim[2], self$n_heads, -1))$permute(c(3,1,2,4))

    dots_text <- torch::torch_einsum('hblk,hbtk->hblt', list(query_text_nh, key_text_nh)) / self$scale

    # 1D relative positions (query, key)
    rel_pos_embed_text <- self$relative_positions_text(seq_length, seq_length)
    rel_pos_key_text <- torch::torch_einsum('bhrd,lrd->bhlr', list(key_text_nh, rel_pos_embed_text))
    rel_pos_query_text <- torch::torch_einsum('bhld,lrd->bhlr', list(query_text_nh, rel_pos_embed_text))

    # shared spatial <-> text hidden features
    key_spatial_text <- self$fc_k_spatial(text_spatial_feat)$unsqueeze(4)
    dim <- key_spatial_text$shape
    key_spatial_text_nh <- key_spatial_text$reshape(c(dim[1], dim[2], self$n_heads, -1))$permute(c(3,1,2,4)) # 'b t (head k) -> head b t k'
    query_spatial_text <- self$fc_q_spatial(text_spatial_feat)$unsqueeze(4)
    dim <- query_spatial_text$shape
    query_spatial_text_nh <- query_spatial_text$reshape(c(dim[1], dim[2], self$n_heads, -1))$permute(c(3,1,2,4))  # 'b l (head k) -> head b l k'
    dots_text_spatial <- torch::torch_einsum('hblk,hbtk->hblt', list(query_spatial_text_nh, key_spatial_text_nh)) / self$scale

    # Line 38 of pseudo-code
    text_attn_scores <- dots_text + rel_pos_key_text + rel_pos_query_text + dots_text_spatial

    # self-attention of image
    key_img_bthk <- self$fc_k_img(img_feat)$unsqueeze(4)
    dim <- key_img_bthk$shape
    key_img_nh <- key_img_bthk$reshape(c(dim[1], dim[2], self$n_heads, -1))$permute(c(3,1,2,4)) # 'b t (head k) -> head b t k'
    query_img_blhk <- self$fc_q_img(img_feat)$unsqueeze(4)
    dim <- query_img_blhk$shape
    query_img_nh <- query_img_blhk$reshape(c(dim[1], dim[2], self$n_heads, -1))$permute(c(3,1,2,4)) # 'b l (head k) -> head b l k'
    value_img_bthk <- self$fc_v_img(img_feat)$unsqueeze(4) # 'b t (head k) -> head b t k'
    dim <- value_img_bthk$shape
    value_img_nh <- value_img_bthk$reshape(c(dim[1], dim[2], self$n_heads, -1))$permute(c(3,1,2,4)) # 'b t (head k) -> head b t k'
    dots_img <- torch::torch_einsum('hblk,hbtk->hblt', list(query_img_nh, key_img_nh)) / self$scale

    # 1D relative positions (query, key)
    rel_pos_embed_img <- self$relative_positions_img(seq_length, seq_length)
    rel_pos_key_img <- torch::torch_einsum('bhrd,lrd->bhlr', list(key_img_nh, rel_pos_embed_text))
    rel_pos_query_img <- torch::torch_einsum('bhld,lrd->bhlr', list(query_img_nh, rel_pos_embed_text))

    # shared spatial <-> image features
    key_spatial_img <- self$fc_k_spatial(img_spatial_feat)$unsqueeze(4)
    dim <- key_spatial_img$shape
    key_spatial_img_nh <- key_spatial_img$reshape(c(dim[1], dim[2], self$n_heads, -1))$permute(c(3,1,2,4)) # 'b t (head k) -> head b t k'
    query_spatial_img <- self$fc_q_spatial(img_spatial_feat)$unsqueeze(4)
    dim <- query_spatial_img$shape
    query_spatial_img_nh <- query_spatial_img$reshape(c(dim[1], dim[2], self$n_heads, -1))$permute(c(3,1,2,4)) # 'b l (head k) -> head b l k'
    dots_img_spatial <- torch::torch_einsum('hblk,hbtk->hblt', list(query_spatial_img_nh, key_spatial_img_nh)) / self$scale

    # Line 59 of pseudo-code
    img_attn_scores <- dots_img + rel_pos_key_img + rel_pos_query_img + dots_img_spatial

    text_attn_probs <- self$softmax_dropout(text_attn_scores)
    img_attn_probs <- self$softmax_dropout(img_attn_scores)

    text_context <- torch::torch_einsum('hblt,hbtv->hblv', list(text_attn_probs, value_text_nh))
    img_context <- torch::torch_einsum('hblt,hbtv->hblv', list(img_attn_probs, value_img_nh))

    context <- text_context + img_context
    dim <- context$shape
    embeddings <- context$permute(c(2,3,1,4))$reshape(c(dim[1], dim[2], -1, 1))$squeeze(4) # 'head b t d -> b t (head d)')
    return(self$to_out(embeddings))
  }
)
docformer_encoder <- torch::nn_module(
  "docformer_encoder",
  initialize = function(config){
    self$config <- config
    hidden_size <- config$hidden_size
    self$layers <- torch::nn_module_list()
    for (i in seq(config$num_hidden_layers)){
      encoder_block <- torch::nn_module_list(
        pre_norm_attention(hidden_size,
                    multimodal_attention_layer(hidden_size,
                                             config$num_attention_heads,
                                             config$max_relative_positions,
                                             config$max_position_embeddings,
                                             config$hidden_dropout_prob,
                    )
        ),
        pre_norm(hidden_size,
                feed_forward(hidden_size,
                            hidden_size * config$intermediate_ff_size_factor,
                            dropout=config$hidden_dropout_prob))
      )
    self$layers$append(encoder_block)
    }

  },
  forward = function(text_feat,  # text feat or output from last encoder block
                     img_feat,
                     text_spatial_feat,
                     img_spatial_feat) {
    for (encoder_block in self$layers){
      skip <- text_feat + img_feat + text_spatial_feat + img_spatial_feat
      attn <- encoder_block[[1]]
      ff <- encoder_block[[2]]
      x <- attn(text_feat, img_feat, text_spatial_feat, img_spatial_feat) + skip
      x <- ff(x) + x
      text_feat <- x
      }
    return(x)

  }
)

language_feature_extractor <- torch::nn_module(
  "language_feature_extractor",
  initialize = function(max_vocab_size, hidden_dim){
    self$embedding_vector <- torch::nn_embedding(max_vocab_size,hidden_dim)
  },
  forward = function() {
    return(self$embedding_vector(x))
  }
)
extract_features <- torch::nn_module(
  "extract_features",
  initialize = function(config){
    self$visual_feature <- resnet_feature_extractor()
    self$language_feature <- language_feature_extractor(config$vocab_size, config$hidden_size)
    self$spatial_feature <- docformer_embeddings(config)

  },
  forward = function(encoding) {
    image <- encoding$resized_scaled_img
    language <- encoding$input_ids
    x_feature <- encoding$x_features
    y_feature <- encoding$y_features

    v_bar <- self$visual_feature(image)
    t_bar <- self$language_feature(language)
    v_bar_s_t_bar_s <- self$spatial_feature(x_feature, y_feature)
    return(list(v_bar, t_bar, v_bar_s_t_bar_s[[1]], v_bar_s_t_bar_s[[2]]))

  }
)

docformer_for_classification <- torch::nn_module(
  "docformer_for_classification",
  initialize = function(config, num_classes){
    self$config <- config
    self$extract_feature <- extract_features(config)
    self$encoder <- docformer_encoder(config)
    self$dropout <- torch::nn_dropout(config$hidden_dropout_prob)
    self$classifier <- torch::nn_linear(hidden_size, num_classes)

  },
  forward = function() {
    output <- self$extract_feature(x) %>%
      self$encoder() %>%
      self$dropout(output) %>%
      self$classifier(output)
  }
)

docformer_for_ir <- torch::nn_module(
  "docformer_for_ir",
  initialize = function(config,num_classes= 30522){
    self$config <- config
    self$extract_feature <- extract_features(config)
    self$encoder <- docformer_encoder(config)
    self$dropout <- torch::nn_dropout(config$hidden_dropout_prob)
    self$classifier <- torch::nn_linear(in_features=68, out_features=num_classes)
    self$decoder <- shallow_decoder()$cuda()
  },
  forward = function() {
    output <- self$extract_feature(x) %>%
      self$encoder() %>%
      self$dropout(output)
    output_mlm <- self$classifier(output)
    output_ir <- self$decoder(output)
    return(list('mlm_labels'=output_mlm,'ir'=output_ir))

  }
)

docformer <- torch::nn_module(
  "docformer",
  initialize = function(config){
    self$config <- config
    self$extract_feature <- extract_features(config)
    self$encoder <- docformer_encoder(config)
    self$dropout <- torch::nn_dropout(config$hidden_dropout_prob)

  },
  forward = function() {
    output <- self$extract_feature(x) %>%
      self$encoder() %>%
      self$dropout(output)

  }
)

shallow_decoder <- torch::nn_module(
  "shallow_decoder",
  initialize = function(){
    self$linear1 <- torch::nn_Linear(in_features = 768, out_features = 512)                        # Making the image to be symmetric
    self$bn0 <- torch::nn_batch_norm2d(num_features = 1)
    self$conv1 <- torch::nn_Conv2d(in_channels = 1, out_channels = 3, kernel_size = 3, stride = 1)
    self$bn1 <- torch::nn_batch_norm2d(num_features = 3)
    self$conv2 <- torch::nn_Conv2d(in_channels = 3, out_channels = 3, kernel_size = 3, stride = 1)
    self$bn2 <- torch::nn_batch_norm2d(num_features = 3)
    self$conv3 <- torch::nn_Conv2d(in_channels = 3, out_channels = 3, kernel_size = 5, stride = 1)
    self$bn3 <- torch::nn_batch_norm2d(num_features = 3)
    self$conv4 <- torch::nn_Conv2d(in_channels = 3, out_channels = 3, kernel_size = 5, stride = 2)
    self$bn4 <- torch::nn_batch_norm2d(num_features = 3)

    self$conv5 <- torch::nn_Conv2d(in_channels = 3, out_channels = 3, kernel_size = 7)
    self$bn5 <- torch::nn_batch_norm2d(num_features = 3)
    self$conv6 <- torch::nn_Conv2d(in_channels = 3, out_channels = 3, kernel_size = 7)
    self$bn6 <- torch::nn_batch_norm2d(num_features = 3)
    self$conv7 <- torch::nn_Conv2d(in_channels = 3, out_channels = 3, kernel_size = 7)
    self$bn7 <- torch::nn_batch_norm2d(num_features = 3)
    self$conv8 <- torch::nn_Conv2d(in_channels = 3, out_channels = 3, kernel_size = 7)
    self$bn8 <- torch::nn_batch_norm2d(num_features = 3)
    self$conv9 <- torch::nn_Conv2d(in_channels = 3, out_channels = 3, kernel_size = 3, stride = 1)
    self$bn9 <- torch::nn_batch_norm2d(num_features = 3)

  },
  forward = function(x) {
    x$unsqueeze(2) %>%
      self$linear1() %>% self$bn0() %>% torch::nn_relu() %>%
      self$conv1()%>% self$bn1() %>% torch::nn_relu() %>%
      self$conv2()%>% self$bn2() %>% torch::nn_relu() %>%
      self$conv3()%>% self$bn3() %>% torch::nn_relu() %>%
      self$conv4()%>% self$bn4() %>% torch::nn_relu() %>%
      self$conv5()%>% self$bn5() %>% torch::nn_relu() %>%
      self$conv6()%>% self$bn6() %>% torch::nn_relu() %>%
      self$conv7()%>% self$bn7() %>% torch::nn_relu() %>%
      self$conv8()%>% self$bn8() %>% torch::nn_relu() %>%
      self$conv9()%>% self$bn9() %>% torch::nn_relu()
  }
)

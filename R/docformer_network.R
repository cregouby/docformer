positional_encoding <- torch::nn_module(
  "positional_encoding",
  initialize = function(d_model, dropout=0.1, max_len=5000){
    self$dropout <- torch::nn_dropout(p=dropout)
    self$max_len <- max_len
    self$d_model <- d_model
    position <- torch::torch_arange(start=1, end = max_len)$unsqueeze(2)
    div_term <- torch::torch_exp(torch::torch_arange(1, d_model, 2) * (-log(1e5)/d_model))
    pe <- torch::torch_zeros(1, max_len, d_model, device = self$config$device)
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
    # use ResNet model for visual features embedding (remove classificaion head)
    # extract resnet50 `layer 4`
    # TODO swith to torch::nn_prune_head as soon as in cran
    self$resnet50 <- .prune_head(torchvision::model_resnet50(pretrain=TRUE))
    # Applying convolution and linear layer
    self$conv1 <- torch::nn_conv2d(2048,768,kernel_size=1)
    self$relu1 <- torch::nn_relu()
    self$linear1 <- torch::nn_linear(768,512)
  },
  forward = function(x) {
    x  <- self$resnet50(x)
    x  <- self$conv1(x)
    x  <- self$relu1(x)
    y  <- x$reshape(c(x$shape[1:2], -1)) # "b e w h -> b e (w.h)" batch, embedding, w, h
    y  <- y$movedim(1,2) # "b e s -> b s e", batch, embedding, sequence
    y  <- self$linear1(y)
    return(y)
  }
)
docformer_embeddings <- torch::nn_module(
  "docformer_embeddings",
  initialize = function(config){
    self$config <- config
    max_2d_p_emb <- config$max_2d_position_embeddings
    rel_max_2d_p_emb <- 2 * max_2d_p_emb + 1

    # self$word_embedding <- torch::nn_embedding(config$vocab_size, config$hidden_size, padding_idx = config$pad_token_id)
    self$position_embedding_v <- positional_encoding(d_model=config$hidden_size, dropout=0.1, max_len=config$max_position_embeddings)

    self$x_topleft_position_embeddings_v <- torch::nn_embedding(max_2d_p_emb, config$coordinate_size)
    self$x_bottomright_position_embeddings_v <- torch::nn_embedding(max_2d_p_emb, config$coordinate_size)
    self$w_position_embeddings_v <- torch::nn_embedding(max_2d_p_emb, config$shape_size)
    self$x_topleft_distance_to_prev_embeddings_v <- torch::nn_embedding(rel_max_2d_p_emb, config$shape_size)
    self$x_bottomright_distance_to_prev_embeddings_v <- torch::nn_embedding(rel_max_2d_p_emb, config$shape_size)
    self$x_centroid_distance_to_prev_embeddings_v <- torch::nn_embedding(rel_max_2d_p_emb, config$shape_size)

    self$y_topleft_position_embeddings_v <- torch::nn_embedding(max_2d_p_emb, config$coordinate_size)
    self$y_bottomright_position_embeddings_v <- torch::nn_embedding(max_2d_p_emb, config$coordinate_size)
    self$h_position_embeddings_v <- torch::nn_embedding(max_2d_p_emb, config$shape_size)
    self$y_topleft_distance_to_prev_embeddings_v <- torch::nn_embedding(rel_max_2d_p_emb, config$shape_size)
    self$y_bottomright_distance_to_prev_embeddings_v <- torch::nn_embedding(rel_max_2d_p_emb, config$shape_size)
    self$y_centroid_distance_to_prev_embeddings_v <- torch::nn_embedding(rel_max_2d_p_emb, config$shape_size)

    self$position_embedding_t <- positional_encoding(d_model=config$hidden_size, dropout=0.1, max_len=config$max_position_embeddings)

    self$x_topleft_position_embeddings_t <- torch::nn_embedding(max_2d_p_emb, config$coordinate_size)
    self$x_bottomright_position_embeddings_t <- torch::nn_embedding(max_2d_p_emb, config$coordinate_size)
    self$w_position_embeddings_t <- torch::nn_embedding(max_2d_p_emb, config$shape_size)
    self$x_topleft_distance_to_prev_embeddings_t <- torch::nn_embedding(rel_max_2d_p_emb, config$shape_size)
    self$x_bottomright_distance_to_prev_embeddings_t <- torch::nn_embedding(rel_max_2d_p_emb, config$shape_size)
    self$x_centroid_distance_to_prev_embeddings_t <- torch::nn_embedding(rel_max_2d_p_emb, config$shape_size)

    self$y_topleft_position_embeddings_t <- torch::nn_embedding(max_2d_p_emb, config$coordinate_size)
    self$y_bottomright_position_embeddings_t <- torch::nn_embedding(max_2d_p_emb, config$coordinate_size)
    self$h_position_embeddings_t <- torch::nn_embedding(max_2d_p_emb, config$shape_size)
    self$y_topleft_distance_to_prev_embeddings_t <- torch::nn_embedding(rel_max_2d_p_emb, config$shape_size)
    self$y_bottomright_distance_to_prev_embeddings_t <- torch::nn_embedding(rel_max_2d_p_emb, config$shape_size)
    self$y_centroid_distance_to_prev_embeddings_t <- torch::nn_embedding(rel_max_2d_p_emb, config$shape_size)

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
    # What are the 6 features:
    #
    # 1 -> top left x/y
    # 2 -> bottom right x/y
    # 3 -> width/height
    # 4 -> diff top left x/y
    # 5 -> diff bottom right x/y
    # 6 -> centroids diff x/y

    batch <- x_feature.shape(1)
    seq_len  <-  x_feature.shape(2)
    num_feat  <-  x_feature.shape(3) # 6
    hidden_size  <-  self$config$hidden_size
    sub_dim  <-  hidden_size %/% num_feat

    # Clamp and add a bias for handling negative values
    x_feature[,,4:N] <- torch::torch_clamp(x_feature[,,4:N],-self$config["max_2d_position_embeddings"],self$config["max_2d_position_embeddings"])+self$config["max_2d_position_embeddings"]
    y_feature[,,4:N] <- torch::torch_clamp(y_feature[,,4:N],-self$config["max_2d_position_embeddings"],self$config["max_2d_position_embeddings"])+self$config["max_2d_position_embeddings"]

    x_topleft_position_embeddings_v <- self$x_topleft_position_embeddings_v(x_feature[,,1])
    x_bottomright_position_embeddings_v <- self$x_bottomright_position_embeddings_v(x_feature[,,2])
    w_position_embeddings_v <- self$w_position_embeddings_v(x_feature[,,3])
    x_topleft_distance_to_prev_embeddings_v <- self$x_topleft_distance_to_prev_embeddings_v(x_feature[,,4])
    x_bottomright_distance_to_prev_embeddings_v <- self$x_bottomright_distance_to_prev_embeddings_v(x_feature[,,5])
    x_centroid_distance_to_prev_embeddings_v <- self$x_centroid_distance_to_prev_embeddings_v(x_feature[,,6])

    x_calculated_embedding_v <- torch::torch_cat(
      c(x_topleft_position_embeddings_v,
        x_bottomright_position_embeddings_v,
        w_position_embeddings_v,
        x_topleft_distance_to_prev_embeddings_v,
        x_bottomright_distance_to_prev_embeddings_v ,
        x_centroid_distance_to_prev_embeddings_v),
      dim = -1
    )

    y_topleft_position_embeddings_v <- self$y_topleft_position_embeddings_v(y_feature[,,1])
    y_bottomright_position_embeddings_v <- self$y_bottomright_position_embeddings_v(y_feature[,,2])
    h_position_embeddings_v <- self$h_position_embeddings_v(y_feature[,,3])
    y_topleft_distance_to_prev_embeddings_v <- self$y_topleft_distance_to_prev_embeddings_v(y_feature[,,4])
    y_bottomright_distance_to_prev_embeddings_v <- self$y_bottomright_distance_to_prev_embeddings_v(y_feature[,,5])
    y_centroid_distance_to_prev_embeddings_v <- self$y_centroid_distance_to_prev_embeddings_v(y_feature[,,6])

    x_calculated_embedding_v <- torch::torch_cat(
      c(x_topleft_position_embeddings_v,
        x_bottomright_position_embeddings_v,
        w_position_embeddings_v,
        x_topleft_distance_to_prev_embeddings_v,
        x_bottomright_distance_to_prev_embeddings_v ,
        x_centroid_distance_to_prev_embeddings_v),
      dim = -1
    )

    y_calculated_embedding_v <- torch::torch_cat(
      c(y_topleft_position_embeddings_v,
        y_bottomright_position_embeddings_v,
        h_position_embeddings_v,
        y_topleft_distance_to_prev_embeddings_v,
        y_bottomright_distance_to_prev_embeddings_v ,
        y_centroid_distance_to_prev_embeddings_v),
    dim = -1
    )
    v_bar_s <-  x_calculated_embedding_v + y_calculated_embedding_v + self.position_embeddings_v()


    x_topleft_position_embeddings_t <- self$x_topleft_position_embeddings_t(x_feature[,,1])
    x_bottomright_position_embeddings_t <- self$x_bottomright_position_embeddings_t(x_feature[,,2])
    w_position_embeddings_t <- self$w_position_embeddings_t(x_feature[,,3])
    x_topleft_distance_to_prev_embeddings_t <- self$x_topleft_distance_to_prev_embeddings_t(x_feature[,,4])
    x_bottomright_distance_to_prev_embeddings_t <- self$x_bottomright_distance_to_prev_embeddings_t(x_feature[,,5])
    x_centroid_distance_to_prev_embeddings_t <- self$x_centroid_distance_to_prev_embeddings_t(x_feature[,,6])

    x_calculated_embedding_t <- torch::torch_cat(
      c(x_topleft_position_embeddings_t,
        x_bottomright_position_embeddings_t,
        w_position_embeddings_t,
        x_topleft_distance_to_prev_embeddings_t,
        x_bottomright_distance_to_prev_embeddings_t ,
        x_centroid_distance_to_prev_embeddings_t),
      dim = -1
    )

    y_topleft_position_embeddings_t <- self$y_topleft_position_embeddings_t(y_feature[,,1])
    y_bottomright_position_embeddings_t <- self$y_bottomright_position_embeddings_t(y_feature[,,2])
    h_position_embeddings_t <- self$h_position_embeddings_t(y_feature[,,3])
    y_topleft_distance_to_prev_embeddings_t <- self$y_topleft_distance_to_prev_embeddings_t(y_feature[,,4])
    y_bottomright_distance_to_prev_embeddings_t <- self$y_bottomright_distance_to_prev_embeddings_t(y_feature[,,5])
    y_centroid_distance_to_prev_embeddings_t <- self$y_centroid_distance_to_prev_embeddings_t(y_feature[,,6])

    x_calculated_embedding_t <- torch::torch_cat(
      c(x_topleft_position_embeddings_t,
        x_bottomright_position_embeddings_t,
        w_position_embeddings_t,
        x_topleft_distance_to_prev_embeddings_t,
        x_bottomright_distance_to_prev_embeddings_t ,
        x_centroid_distance_to_prev_embeddings_t),
      dim = -1
    )

    y_calculated_embedding_t <- torch::torch_cat(
      c(y_topleft_position_embeddings_t,
        y_bottomright_position_embeddings_t,
        h_position_embeddings_t,
        y_topleft_distance_to_prev_embeddings_t,
        y_bottomright_distance_to_prev_embeddings_t ,
        y_centroid_distance_to_prev_embeddings_t),
      dim = -1
    )

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
    self$embeddings_table <- torch::nn_parameter(torch::torch_zeros(c(max_relative_position *2 +1, num_units), device=self$config$device))
    torch::nn_init_xavier_uniform_(self$embeddings_table)

    self$max_length <- max_seq_length
    range_vec_q  <-  torch::torch_arange(1, max_seq_length)
    range_vec_k  <-  torch::torch_arange(1, max_seq_length)
    distance_mat  <-  range_vec_k[NULL, ..] - range_vec_q[.., NULL]
    distance_mat_clipped  <-  torch::torch_clamp(distance_mat, -self$max_relative_position, self$max_relative_position)
    final_mat  <-  distance_mat_clipped + self$max_relative_position
    self$final_mat  <-  final_mat$to(dtype=torch::torch_long())
  },
  forward = function(length_q, length_k) {
    embeddings  <-  self$embeddings_table[self$final_mat[1:length_q, 1:length_k]]
  }
)

multimodal_attention_layer <- torch::nn_module(
  "multimodal_attention_layer",
  initialize = function(embed_dim, n_heads, max_relative_position, max_seq_length, dropout){
    self$embed_dim <- embed_dim
    self$n_heads <- n_heads
    self$head_dim <- embed_dim %/% n_heads

    self$relative_positions_text <- relative_position(self$head_dim, max_relative_position, max_seq_length)
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
      torch::nn_softmax(dim=-1),
      torch::nn_dropout(dropout)
    )

    self$to_out <- torch::nn_sequential(
      torch::nn_linear(embed_dim, embed_dim),
      torch::nn_dropout(dropout)
    )
    self$scale <- torch::torch_sqrt(torch::torch_tensor(embed_dim, device = self$config$device))


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
      encoder_block <- torch::nn_module_list(list(
        pre_norm_attention(hidden_size,
                    multimodal_attention_layer(hidden_size,
                                             config$num_attention_heads,
                                             config$max_relative_positions,
                                             config$max_position_embeddings,
                                             config$hidden_dropout_prob
                    )
        ),
        pre_norm(hidden_size,
                feed_forward(hidden_size,
                            config$intermediate_size,
                            dropout=config$hidden_dropout_prob))
      ))
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
  initialize = function(config){
    layoutlm_net <- LayoutLMForTokenClassification(config)$from_pretrained("microsoft/layoutlm-base-uncased")
    self$embedding_vector <- torch::nn_embedding(config$vocab_size, config$hidden_size, .weight=layoutlm_net$layoutlm$embeddings$word_embeddings$weight)
  },
  forward = function(x) {
    return(self$embedding_vector(x))
  }
)
extract_features <- torch::nn_module(
  "extract_features",
  initialize = function(config){
    self$visual_feature <- resnet_feature_extractor()
    self$language_feature <- language_feature_extractor(config)
    self$spatial_feature <- docformer_embeddings(config)

  },
  forward = function(encoding) {
    image <- encoding$image
    language <- encoding$text
    x_feature <- encoding$x_features
    y_feature <- encoding$y_features

    v_bar <- self$visual_feature(image)
    t_bar <- self$language_feature(language)
    v_bar_s_t_bar_s <- self$spatial_feature(x_feature, y_feature)
    return(list(v_bar, t_bar, v_bar_s_t_bar_s[[1]], v_bar_s_t_bar_s[[2]]))

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
  forward = function(x) {
    x_ex_fe <- self$extract_feature(x)
    x_enc <- self$encoder(x_ex_fe)
    output <- self$dropout(x_enc)

  }
)

LAYOUTLM_PRETRAINED_MODEL_ARCHIVE_LIST <- c("layoutlm-base-uncased","layoutlm-large-uncased")

ACT2FN = list(
  "relu"= torch::nn_relu,
  "swish"= torch::nn_hardswish,
  "gelu"= torch::nn_gelu,
  "tanh"= torch::nn_tanh
)

#'Construct the embeddings from word, position and token_type embeddings.
LayoutLMEmbeddings <- torch::nn_module(
  "LayoutLMEmbeddings",
  initialize = function(config){
    self$word_embeddings <- torch::nn_embedding(config$vocab_size, config$hidden_size, padding_idx=config$pad_token_id)
    self$position_embeddings <- torch::nn_embedding(config$max_position_embeddings, config$hidden_size)
    self$x_position_embeddings <- torch::nn_embedding(config$max_2d_position_embeddings, config$hidden_size)
    self$y_position_embeddings <- torch::nn_embedding(config$max_2d_position_embeddings, config$hidden_size)
    self$h_position_embeddings <- torch::nn_embedding(config$max_2d_position_embeddings, config$hidden_size)
    self$w_position_embeddings <- torch::nn_embedding(config$max_2d_position_embeddings, config$hidden_size)
    self$token_type_embeddings <- torch::nn_embedding(config$type_vocab_size, config$hidden_size)
    
    self$LayerNorm <- torch::nn_layer_norm(config$hidden_size, eps=config$layer_norm_eps)
    self$dropout <- torch::nn_dropout(config$hidden_dropout_prob)
    
    # self$register_buffer("position_ids", torch::torch_arange(config$max_position_embeddings)$expand(c(1, -1)))
    self$position_ids <-  torch::torch_arange(start=1, end=config$max_position_embeddings)$expand(c(1, -1))
  },
  forward = function(
    input_ids=NULL,
    bbox=NULL,
    token_type_ids=NULL,
    position_ids=NULL,
    inputs_embeds=NULL
  ){
    if (!is.null(input_ids)){
      input_shape <- input_ids$size()
    } else {
      input_shape <- inputs_embeds$shape[c(1:(inputs_embeds$ndim-1))]
    }
    seq_length <- input_shape[2]
    
    device <- ifelse(!is.null(input_ids), input_ids$device, inputs_embeds$device)
    
    if (is.null(position_ids)){
      position_ids <- self$position_ids[, 1:(seq_length-1)]
    }
    
    if (is.null(token_type_ids)){
      token_type_ids <- torch::torch_zeros(input_shape, dtype=torch::torch_long(), device=device)
    }
    
    if (is.null(inputs_embeds)){
      inputs_embeds <- self$word_embeddings(input_ids)
    }
    
    words_embeddings <- inputs_embeds
    position_embeddings <- self$position_embeddings(position_ids)
    left_position_embeddings <- self$x_position_embeddings(bbox[, , 1])
    upper_position_embeddings <- self$y_position_embeddings(bbox[, , 2])
    right_position_embeddings <- self$x_position_embeddings(bbox[, , 3])
    lower_position_embeddings <- self$y_position_embeddings(bbox[, , 4])
    
    h_position_embeddings <- self$h_position_embeddings(bbox[, , 4] - bbox[, , 2])
    w_position_embeddings <- self$w_position_embeddings(bbox[, , 3] - bbox[, , 1])
    token_type_embeddings <- self$token_type_embeddings(token_type_ids)
    
    embeddings <- torch::torch_cat(
      words_embeddings,
      position_embeddings,
      left_position_embeddings,
      upper_position_embeddings,
      right_position_embeddings,
      lower_position_embeddings,
      h_position_embeddings,
      w_position_embeddings,
      token_type_embeddings
    )
    # TODO check if LayoutLMLayerNorm is managing that tensor shape right
    embeddings <- self$LayerNorm(embeddings)
    embeddings <- self$dropout(embeddings)
    return(embeddings)
  }
)

# Copied from transformers$models$bert$modeling_bert$BertSelfAttention with Bert->LayoutLM
LayoutLMSelfAttention <- torch::nn_module(
  "LayoutLMSelfAttention",
  initialize = function(config, position_embedding_type=NULL){
    stopifnot("The config hidden size is not a multiple of the number of attention heads" = (config$hidden_size %% config$num_attention_heads == 0))
    
    self$num_attention_heads <- config$num_attention_heads
    self$attention_head_size <- config$hidden_size %/% config$num_attention_heads
    self$all_head_size <- self$num_attention_heads * self$attention_head_size
    
    self$query <- torch::nn_linear(config$hidden_size, self$all_head_size)
    self$key <- torch::nn_linear(config$hidden_size, self$all_head_size)
    self$value <- torch::nn_linear(config$hidden_size, self$all_head_size)
    
    self$dropout <- torch::nn_dropout(p=config$attention_dropout_prob)
    # get first non null within position_embedding_type, config$position_embedding_type, "absolute"
    self$position_embedding_type <- (append(position_embedding_type,config$position_embedding_type) %>% append("absolute"))[[1]]
    if (self$position_embedding_type %in% c("relative_key","relative_key_query")){
      self$max_position_embeddings <- config$max_position_embeddings
      self$distance_embedding <- torch::nn_embedding(2 * config$max_position_embeddings - 1, self$attention_head_size)
    }
    
    self$is_decoder <- config$is_decoder
  },
  transpose_for_scores = function(x){
    new_x_shape <- append(x$shape[1:(x$ndim-1)], self$num_attention_heads, self$attention_head_size)
    x <- x$view(new_x_shape)
    x$permute(1, 3, 2, 4)
  },
  forward = function(
    
    hidden_states,
    attention_mask=NULL,
    head_mask=NULL,
    encoder_hidden_states=NULL,
    encoder_attention_mask=NULL,
    past_key_value=NULL,
    output_attentions=FALSE
  ){
    mixed_query_layer <- self$query(hidden_states)
    
    # If this is instantiated as a cross-attention module, the keys
    # and values come from an encoder; the attention mask needs to be
    # such that the encoder's padding tokens are not attended to.
    is_cross_attention <- !is.null(encoder_hidden_states)
    
    if (is_cross_attention & !is.null(past_key_value)){
      # reuse k,v, cross_attentions
      key_layer <- past_key_value[1]
      value_layer <- past_key_value[2]
      attention_mask <- encoder_attention_mask
    } else if (is_cross_attention){
      key_layer <- self$transpose_for_scores(self$key(encoder_hidden_states))
      value_layer <- self$transpose_for_scores(self$value(encoder_hidden_states))
      attention_mask <- encoder_attention_mask
    } else if (!is.null(past_key_value)){
      key_layer <- self$transpose_for_scores(self$key(hidden_states))
      value_layer <- self$transpose_for_scores(self$value(hidden_states))
      key_layer <- torch::torch_cat(list(past_key_value[1], key_layer), dim=2)
      value_layer <- torch::torch_cat(list(past_key_value[2], value_layer), dim=2)
    } else {
      key_layer <- self$transpose_for_scores(self$key(hidden_states))
      value_layer <- self$transpose_for_scores(self$value(hidden_states))
    }
    
    query_layer <- self$transpose_for_scores(mixed_query_layer)
    
    if (self$is_decoder){
      # if cross_attention save Tuple(torch::torch_Tensor, torch::torch_Tensor) of all cross attention key/value_states.
      # Further calls to cross_attention layer can then reuse all cross-attention
      # key/value_states (first "if" case)
      # if uni-directional self-attention (decoder) save Tuple(torch::torch_Tensor, torch::torch_Tensor) of
      # all previous decoder key/value_states. Further calls to uni-directional self-attention
      # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
      # if encoder bi-directional self-attention `past_key_value` is always `NULL`
      past_key_value <- list(key_layer, value_layer)
    }
    # Take the dot product between "query" and "key" to get the raw attention scores.
    attention_scores <- torch::torch_matmul(query_layer, key_layer$transpose(-1, -2))
    
    if (self$position_embedding_type %in% c("relative_key","relative_key_query")){
      seq_length <- hidden_states$size()[2]
      position_ids_l <- torch::torch_arange(seq_length, dtype=torch::torch_long, device=hidden_states$device)$view(c(-1, 1))
      position_ids_r <- torch::torch_arange(seq_length, dtype=torch::torch_long, device=hidden_states$device)$view(c(1, -1))
      distance <- position_ids_l - position_ids_r
      positional_embedding <- self$distance_embedding(distance + self$max_position_embeddings - 1)
      positional_embedding <- positional_embedding$to(dtype=query_layer$dtype)  # fp16 compatibility
      
      if (self$position_embedding_type == "relative_key"){
        relative_position_scores <- torch::torch_einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
        attention_scores <- attention_scores + relative_position_scores
      } else if (self$position_embedding_type == "relative_key_query"){
        relative_position_scores_query <- torch::torch_einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
        relative_position_scores_key <- torch::torch_einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
        attention_scores <- attention_scores + relative_position_scores_query + relative_position_scores_key
      }
    }
    attention_scores <- attention_scores / sqrt(self$attention_head_size)
    if (!is.null(attention_mask)){
      # Apply the attention mask is (precomputed for all layers in LayoutLMModel forward() function)
      attention_scores <- attention_scores + attention_mask
    }
    # Normalize the attention scores to probabilities.
    attention_probs <- torch::nnf_softmax(attention_scores, dim=-1)
    
    # This is actually dropping out entire tokens to attend to, which might
    # seem a bit unusual, but is taken from the original Transformer paper.
    attention_probs <- self$dropout(attention_probs)
    
    # Mask heads if we want to
    if (!is.null(head_mask)){
      attention_probs <- attention_probs * head_mask
    }
    context_layer <- torch::torch_matmul(attention_probs, value_layer)
    
    context_layer <- context_layer$permute(c(1, 3, 2, 4))$contiguous()
    # TODO is it broadcasting second term or ?
    new_context_layer_shape <- append(context_layer$shape[1:(context_layer$ndim-2)],self$all_head_size)
    context_layer <- context_layer$view(new_context_layer_shape)
    
    
    out_lst <- list(context_layer)
    if (output_attentions){
      out_lst <- append(out_lst, attention_probs)
    }
    if (self$is_decoder){
      out_lst <- append(out_lst, past_key_value)
    }
    return(out_lst)
  }
)

# Copied from transformers$models$bert$modeling_bert$BertSelfOutput with Bert->LayoutLM
LayoutLMSelfOutput <- torch::nn_module(
  "LayoutLMSelfOutput",
  initialize = function(config){
    self$dense <- torch::nn_linear(config$hidden_size, config$hidden_size)
    self$LayerNorm <- torch::nn_layer_norm(config$hidden_size, eps=config$layer_norm_eps)
    self$dropout <- torch::nn_dropout(config$hidden_dropout_prob)
  },
  forward = function(hidden_states, input_tensor){
    hidden_states <- self$dense(hidden_states)
    hidden_states <- self$dropout(hidden_states)
    hidden_states <- self$LayerNorm(hidden_states + input_tensor)
    return(hidden_states)
  }
)

# Copied from transformers$models$bert$modeling_bert$BertAttention with Bert->LayoutLM
LayoutLMAttention <- torch::nn_module(
  "LayoutLMAttention",
  initialize = function(config, position_embedding_type=NULL){
    self$self <- LayoutLMSelfAttention(config, position_embedding_type=position_embedding_type)
    self$output <- LayoutLMSelfOutput(config)
    self$pruned_heads <- list()
  },
  prune_heads = function(heads){
    if (is.null(heads)){
      return
    }
    heads_index_lst <- find_pruneable_heads_and_indices(
      heads, self$self$num_attention_heads, self$self$attention_head_size, self$pruned_heads
    )
    heads <-  heads_index_lst[[1]]
    index <- heads_index_lst[[2]]
    # Prune linear layers
    self$self$query <- prune_linear_layer(self$self$query, index)
    self$self$key <- prune_linear_layer(self$self$key, index)
    self$self$value <- prune_linear_layer(self$self$value, index)
    self$output$dense <- prune_linear_layer(self$output$dense, index, dim=1)
    
    # Update hyper params and store pruned heads
    self$self$num_attention_heads <- self$self$num_attention_heads - len(heads)
    self$self$all_head_size <- self$self$attention_head_size * self$self$num_attention_heads
    self$pruned_heads <- self$pruned_heads$union(heads)
    
  },
  forward = function(
    
    hidden_states,
    attention_mask=NULL,
    head_mask=NULL,
    encoder_hidden_states=NULL,
    encoder_attention_mask=NULL,
    past_key_value=NULL,
    output_attentions=FALSE
  ){
    self_outputs <- self$self(
      hidden_states,
      attention_mask,
      head_mask,
      encoder_hidden_states,
      encoder_attention_mask,
      past_key_value,
      output_attentions
    )
    attention_output <- self$output(self_outputs[[1]], hidden_states)
    out_lst <- append(attention_output, self_outputs[c(2:N)])  # add attentions if we output them
    return(out_lst)
  }
)

# Copied from transformers$models$bert$modeling_bert$BertIntermediate
LayoutLMIntermediate <- torch::nn_module(
  "LayoutLMIntermediate",
  initialize = function(config){
    self$dense <- torch::nn_linear(config$hidden_size, config$intermediate_size)
    if (is.character(config$hidden_act)){
      self$intermediate_act_fn <- ACT2FN[config$hidden_act]
    } else {
      self$intermediate_act_fn <- config$hidden_act
    }
  },
  forward = function(hidden_states){
    hidden_states <- self$dense(hidden_states)
    hidden_states <- self$intermediate_act_fn(hidden_states)
    return(hidden_states)
  }
)

# Copied from transformers$models$bert$modeling_bert$BertOutput with Bert->LayoutLM
LayoutLMOutput <- torch::nn_module(
  "LayoutLMOutput",
  initialize = function(config){
    self$dense <- torch::nn_linear(config$intermediate_size, config$hidden_size)
    self$LayerNorm <- torch::nn_layer_norm(config$hidden_size, eps=config$layer_norm_eps)
    self$dropout <- torch::nn_dropout(config$hidden_dropout_prob)
    
  },
  forward = function(hidden_states, input_tensor){
    hidden_states <- self$dense(hidden_states)
    hidden_states <- self$dropout(hidden_states)
    hidden_states <- self$LayerNorm(hidden_states + input_tensor)
    return(hidden_states)
  }
)

# Copied from transformers$models$bert$modeling_bert$BertLayer with Bert->LayoutLM
LayoutLMLayer <- torch::nn_module(
  "LayoutLMLayer",
  initialize = function(config){
    self$chunk_size_feed_forward <- config$chunk_size_feed_forward
    self$seq_len_dim <- 1L
    self$attention <- LayoutLMAttention(config)
    self$is_decoder <- config$is_decoder
    self$add_cross_attention <- config$add_cross_attention
    if (!is.null(self$add_cross_attention)){
      stopifnot("list(self) should be used as a decoder model if cross attention is added" = self$is_decoder)
      self$crossattention <- LayoutLMAttention(config, position_embedding_type="absolute")
    }
    self$intermediate <- LayoutLMIntermediate(config)
    self$output <- LayoutLMOutput(config)
  },
  forward = function(
    hidden_states,
    attention_mask=NULL,
    head_mask=NULL,
    encoder_hidden_states=NULL,
    encoder_attention_mask=NULL,
    past_key_value=NULL,
    output_attentions=FALSE
  ){
    # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
    self_attn_past_key_value <- ifelse(is.null(past_key_value), NULL, past_key_value[1:3])
    self_attention_outputs <- self$attention(
      hidden_states,
      attention_mask,
      head_mask,
      output_attentions=output_attentions,
      past_key_value=self_attn_past_key_value,
    )
    attention_output <- self_attention_outputs[[1]]
    
    # if decoder, the last output is tuple of self-attn cache
    if (self$is_decoder){
      outputs <- self_attention_outputs[c(2:(length(self_attention_outputs)-1))]
      present_key_value <- self_attention_outputs[[length(self_attention_outputs)]]
    } else {
      outputs <- self_attention_outputs[-1]  # add self attentions if we output attention weights
    }
    cross_attn_present_key_value <- NULL
    if (self$is_decoder & !is.null(encoder_hidden_states)){
      stopifnot("If `encoder_hidden_states` are passed, list(self) has to be instantiated with cross-attention layers by setting `config$add_cross_attention=TRUE`" = !is.null(self$crossattention))
      
      # cross_attn cached key/values tuple is at positions 3,4 of past_key_value tuple
      cross_attn_past_key_value <- ifelse(is.null(past_key_value), NULL, past_key_value[-c(1:(length(past_key_value)-2))])
      cross_attention_outputs <- self$crossattention(
        attention_output,
        attention_mask,
        head_mask,
        encoder_hidden_states,
        encoder_attention_mask,
        cross_attn_past_key_value,
        output_attentions,
      )
      attention_output <- cross_attention_outputs[[1]]
      outputs <- append(outputs,cross_attention_outputs[c(2:(length(cross_attention_outputs)-1))] ) # add cross attentions if we output attention weights
      
      # add cross-attn cache to positions 3,4 of present_key_value tuple
      cross_attn_present_key_value <- cross_attention_outputs[[length(cross_attention_outputs)]]
      present_key_value <- append(present_key_value, cross_attn_present_key_value)
    }
    layer_output <- apply_chunking_to_forward(
      self$feed_forward_chunk, self$chunk_size_feed_forward, self$seq_len_dim, attention_output
    )
    outputs <- append(layer_output,outputs)
    
    # if decoder, return the attn key/values as the last output
    if (self$is_decoder){
      outputs <- append(outputs, present_key_value)
    }
    return(outputs)
    
  },
  feed_forward_chunk = function(attention_output){
    intermediate_output <- self$intermediate(attention_output)
    layer_output <- self$output(intermediate_output, attention_output)
    return(layer_output)
  }
)

# Copied from transformers$models$bert$modeling_bert$BertEncoder with Bert->LayoutLM
LayoutLMEncoder <- torch::nn_module(
  "LayoutLMEncoder",
  initialize = function(config){
    self$config <- config
    self$layer <- torch::nn_module_list(lapply(1:config$num_hidden_layers, function(x) docformer:::LayoutLMLayer(config)))
    self$gradient_checkpointing <- FALSE
    
  },
  forward = function(
    hidden_states,
    attention_mask=NULL,
    head_mask=NULL,
    encoder_hidden_states=NULL,
    encoder_attention_mask=NULL,
    past_key_values=NULL,
    use_cache=NULL,
    output_attentions=FALSE,
    output_hidden_states=FALSE,
    return_dict=TRUE
  ){
    # all_hidden_states <- () if output_hidden_states else NULL
    # all_self_attentions <- () if output_attentions else NULL
    # all_cross_attentions <- () if output_attentions and self$config$add_cross_attention else NULL
    
    # next_decoder_cache <- () if use_cache else NULL
    hidden_layer_seq <- seq(length(self$layer))
    if (output_hidden_states){
      all_hidden_states <- rep(hidden_states, length(self$layer))
      
      
      # layer_head_mask <- head_mask[i] if head_mask is not NULL else NULL
      # past_key_value <- past_key_values[i] if past_key_values is not NULL else NULL
      
      if (self$gradient_checkpointing & self$training){
        if (use_cache==TRUE) {
          rlang::warn("`use_cache=TRUE` is incompatible with gradient checkpointing. Setting `use_cache=FALSE`...")
        }
        use_cache <- FALSE
        layer_outputs <-   purrr::map(
          seq(length(self$layer)),
          ~torch::torch_utils$checkpoint$checkpoint(
            self$layer[[.x]](
              hidden_states,
              attention_mask,
              layer_head_mask,
              encoder_hidden_states,
              encoder_attention_mask,
              past_key_value,
              output_attentions
            )
          )
        )
      } else {
        layer_outputs <-   purrr::map(
          seq(length(self$layer)),
          ~self$layer[[.x]](
            hidden_states,
            attention_mask,
            layer_head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
          ))
      }
      
      all_hidden_states <- purrr::map(layer_outputs, ~.x[[1]])
      if (use_cache){
        next_decoder_cache <- purrr::map(layer_outputs, ~dplyr::last(.x))
      }
      if (output_attentions){
        all_self_attentions <- purrr::map(layer_outputs, ~.x[[2]])
        if (self$config$add_cross_attention){
          all_cross_attentions <- purrr::map(layer_outputs, ~.x[[3]],)
        }
      }
    }
    if (output_hidden_states){
      all_hidden_states <- append(all_hidden_states,hidden_states)
    }
    
    res_lst <- list(
      last_hidden_state = dplyr::last(all_hidden_states),
      past_key_values = next_decoder_cache,
      hidden_states = all_hidden_states,
      attentions = all_self_attentions,
      cross_attentions = all_cross_attentions,
    )
    class(res_lst) <- "BaseModelOutputWithPastAndCrossAttentions"
    return(res_lst)
  }
)


# Copied from transformers$models$bert$modeling_bert$BertPooler
LayoutLMPooler <- torch::nn_module(
  "LayoutLMPooler",
  initialize = function(config){
    self$dense <- torch::nn_linear(config$hidden_size, config$hidden_size)
    self$activation <- torch::nn_tanh()
    
  },
  forward = function(hidden_states){
    # We "pool" the model by simply taking the hidden state corresponding
    # to the first token.
    first_token_tensor <- hidden_states[, 1]
    pooled_output <- self$dense(first_token_tensor)
    pooled_output <- self$activation(pooled_output)
    pooled_output
  }
)

# Copied from transformers$models$bert$modeling_bert$BertPredictionHeadTransform with Bert->LayoutLM
LayoutLMPredictionHeadTransform <- torch::nn_module(
  "LayoutLMPredictionHeadTransform",
  initialize = function(config){
    self$dense <- torch::nn_linear(config$hidden_size, config$hidden_size)
    if (is.character(config$hidden_act)){
      self$transform_act_fn <- ACT2FN[config$hidden_act]
    } else {
      self$transform_act_fn <- config$hidden_act
    }
    self$LayerNorm <- torch::nn_layer_norm(config$hidden_size, eps=config$layer_norm_eps)
    
  },
  forward = function(hidden_states){
    hidden_states <- self$dense(hidden_states)
    hidden_states <- self$transform_act_fn(hidden_states)
    hidden_states <- self$LayerNorm(hidden_states)
    hidden_states
  }
)

# Copied from transformers$models$bert$modeling_bert$BertLMPredictionHead with Bert->LayoutLM
LayoutLMLMPredictionHead <- torch::nn_module(
  "LayoutLMLMPredictionHead",
  initialize = function(config){
    self$transform <- LayoutLMPredictionHeadTransform(config)
    
    # The output weights are the same as the input embeddings, but there is
    # an output-only bias for each token.
    self$decoder <- torch::nn_linear(config$hidden_size, config$vocab_size, bias=FALSE)
    
    self$bias <- torch::nn_parameter(torch::torch_zeros(config$vocab_size))
    
    # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
    self$decoder$bias <- self$bias
    
  },
  forward = function(hidden_states){
    hidden_states <- self$transform(hidden_states)
    hidden_states <- self$decoder(hidden_states)
    hidden_states
  }
)

# Copied from transformers$models$bert$modeling_bert$BertOnlyMLMHead with Bert->LayoutLM
LayoutLMOnlyMLMHead <- torch::nn_module(
  "LayoutLMOnlyMLMHead",
  initialize = function(config){
    self$predictions <- LayoutLMLMPredictionHead(config)
    
  },
  forward = function(sequence_output){
    prediction_scores <- self$predictions(sequence_output)
    prediction_scores
  }
)


#' The LayoutLM model was proposed in [LayoutLM: Pre-training of Text and Layout for Document ImageUnderstanding](https://arxiv.org/abs/1912.13318)
#'  by Yiheng Xu, Minghao Li, Lei Cui, Shaohan Huang, Furu Wei and Ming Zhou.
#'
#' This model is a torch [nn_module()](https://torch.mlverse.org/docs/reference/nn_module.html). Use
#' it as a regular module and refer to the documentation for all matter related to general usage and
#' behavior.
#'
#' Parameters:
#'     config ([`LayoutLMConfig`]): Model configuration class with all the parameters of the model.
#'         Initializing with a config file does not load the weights associated with the model, only the
#'         configuration. Check out the [`~PreTrainedModel$from_pretrained`] method to load the model weights.
LayoutLMModel <- torch::nn_module(
  "LayoutLMPreTrainedModel",
  initialize = function(config){
    self$config <- config
    
    self$embeddings <- LayoutLMEmbeddings(config)
    self$encoder <- LayoutLMEncoder(config)
    self$pooler <- LayoutLMPooler(config)
    
    # Initialize weights and apply final processing
    # self$init_weights()
    
  },
  get_input_embeddings = function(self){
    self$embeddings$word_embeddings
    
  },
  set_input_embeddings = function(value){
    self$embeddings$word_embeddings <- value
    
  },
  prune_heads = function(heads_to_prune){
    # """
    # Prunes heads of the model. heads_to_prune: dict of list(layer_num: list of heads to prune in this layer) See base
    # class PreTrainedModel
    # """
    for (layer_heads_lst in heads_to_prune$items()){
      self$encoder$layer[layer_heads_lst[[1]]]$attention$prune_heads(layer_heads_lst[[2]])
    }
  },
  forward = function(input_ids = NULL,
                     bbox = NULL,
                     attention_mask = NULL,
                     token_type_ids = NULL,
                     position_ids = NULL,
                     head_mask = NULL,
                     inputs_embeds = NULL,
                     encoder_hidden_states = NULL,
                     encoder_attention_mask = NULL,
                     output_attentions = NULL,
                     output_hidden_states = NULL,
                     return_dict = NULL) {
    # -> Unionc(Tuple, BaseModelOutputWithPoolingAndCrossAttentions):
    # r"""
    # Returns:
    #
    # Examples:
    #
    # ```python
    # >>> from transformers import LayoutLMTokenizer, LayoutLMModel
    # >>> import torch
    #
    # >>> tokenizer <- LayoutLMTokenizer$from_pretrained("microsoft/layoutlm-base-uncased")
    # >>> model <- LayoutLMModel$from_pretrained("microsoft/layoutlm-base-uncased")
    #
    # >>> words <- c("Hello", "world")
    # >>> normalized_word_boxes <- [637, 773, 693, 782], [698, 773, 733, 782]
    #
    # >>> token_boxes <- []
    # >>> for word, box in zip(words, normalized_word_boxes){
    # ...     word_tokens <- tokenizer$tokenize(word)
    # ...     token_boxes$extend([box] * len(word_tokens))
    # >>> # add bounding boxes of cls + sep tokens
    # >>> token_boxes <- [[0, 0, 0, 0]] + token_boxes + [[1000, 1000, 1000, 1000]]
    #
    # >>> encoding <- tokenizer(" ".join(words), return_tensors="pt")
    # >>> input_ids <- encoding$input_ids
    # >>> attention_mask <- encoding$attention_mask
    # >>> token_type_ids <- encoding$token_type_ids
    # >>> bbox <- torch::torch_tensor([token_boxes])
    #
    # >>> outputs <- model(
    # ...     input_ids=input_ids, bbox=bbox, attention_mask=attention_mask, token_type_ids=token_type_ids
    # ... )
    #
    # >>> last_hidden_states <- outputs$last_hidden_state
    # ```"""
    output_attentions <- append(output_attentions,self$config$output_attentions)[[1]]
    output_hidden_states <- append(output_hidden_states,self$config$output_hidden_states)[[1]]
    
    return_dict <- append(return_dict,self$config$use_return_dict)[[1]]
    
    if (!is.null(input_ids) & !is.null(inputs_embeds)){
      rlang::abort("You cannot specify both input_ids and inputs_embeds at the same time")
    } else if (!is.null(input_ids)){
      input_shape <- input_ids$size()
    } else if (!is.null(inputs_embeds)){
      input_shape <- inputs_embeds$shape[1:(inputs_embeds$ndim-1)]
    } else {
      rlang::abort("You have to specify either input_ids or inputs_embeds")
    }
    
    device <- ifelse(is.null(input_ids),inputs_embeds$device,input_ids$device)
    
    if (is.null(attention_mask)){
      attention_mask <- torch::torch_ones(input_shape, device=device)
    }
    if (is.null(token_type_ids)){
      token_type_ids <- torch::torch_zeros(input_shape, dtype=torch::torch_long, device=device)
    }
    
    if (is.null(bbox)){
      bbox <- torch::torch_zeros(append(input_shape, 4), dtype=torch::torch_long, device=device)
    }
    
    extended_attention_mask <- attention_mask$unsqueeze(2)$unsqueeze(3)
    
    extended_attention_mask <- extended_attention_mask$to(dtype=self$dtype)
    extended_attention_mask <- (1 - extended_attention_mask) * -10000
    
    if (!is.null(head_mask)){
      if (head_mask$ndim == 1){
        head_mask <- head_mask$unsqueeze(1)$unsqueeze(1)$unsqueeze(-1)$unsqueeze(-1)
        head_mask <- head_mask$expand(self$config$num_hidden_layers, -1, -1, -1, -1)
      } else if (head_mask$ndim == 2){
        head_mask <- head_mask$unsqueeze(2)$unsqueeze(-1)$unsqueeze(-1)
      }
      head_mask <- head_mask$to(dtype=next(self$parameters())$dtype)
    } else {
      head_mask <- list()
    }
    
    embedding_output <- self$embeddings(
      input_ids=input_ids,
      bbox=bbox,
      position_ids=position_ids,
      token_type_ids=token_type_ids,
      inputs_embeds=inputs_embeds,
    )
    encoder_outputs <- self$encoder(
      embedding_output,
      extended_attention_mask,
      head_mask=head_mask,
      output_attentions=output_attentions,
      output_hidden_states=output_hidden_states,
      return_dict=return_dict,
    )
    sequence_output <- encoder_outputs[[1]]
    pooled_output <- self$pooler(sequence_output)
    
    if (!return_dict){
      return(append(sequence_output, pooled_output, encoder_outputs[-1]))
    }
    
    result <-  list(
      last_hidden_state=sequence_output,
      pooler_output=pooled_output,
      hidden_states=encoder_outputs$hidden_states,
      attentions=encoder_outputs$attentions,
      cross_attentions=encoder_outputs$cross_attentions
    )
    class(result) <- "BaseModelOutputWithPoolingAndCrossAttentions"
    return(result)
  }
)

#' LayoutLM Model with a `language modeling` head on top."
LayoutLMForMaskedLM<- torch::nn_module(
  "LayoutLMPreTrainedModel",
  initialize = function(config){
    self$layoutlm <- LayoutLMModel(config)
    self$cls <- LayoutLMOnlyMLMHead(config)
    # Initialize weights and apply final processing
    # self$init_weights()
    
  },
  get_input_embeddings = function(){
    self$layoutlm$embeddings$word_embeddings
    
  },
  get_output_embeddings = function(){
    self$cls$predictions$decoder
    
  },
  set_output_embeddings = function(new_embeddings){
    self$cls$predictions$decoder <- new_embeddings
    
  },
  forward = function(
    input_ids=NULL,
    bbox=NULL,
    attention_mask=NULL,
    token_type_ids=NULL,
    position_ids=NULL,
    head_mask=NULL,
    inputs_embeds=NULL,
    labels=NULL,
    encoder_hidden_states=NULL,
    encoder_attention_mask=NULL,
    output_attentions=NULL,
    output_hidden_states=NULL,
    return_dict=NULL
  ){ # -> Unionc(Tuple, MaskedLMOutput):
    # r"""
    # labels (`torch::torch_LongTensor` of shape `(batch_size, sequence_length)`, *optional*){
    #     Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
    #     config$vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
    #     loss is only computed for the tokens with labels in `[0, ..., config$vocab_size]`
    #
    # Returns:
    #
    # Examples:
    #
    # ```python
    # >>> from transformers import LayoutLMTokenizer, LayoutLMForMaskedLM
    # >>> import torch
    #
    # >>> tokenizer <- LayoutLMTokenizer$from_pretrained("microsoft/layoutlm-base-uncased")
    # >>> model <- LayoutLMForMaskedLM$from_pretrained("microsoft/layoutlm-base-uncased")
    #
    # >>> words <- c("Hello", "[MASK]")
    # >>> normalized_word_boxes <- [637, 773, 693, 782], [698, 773, 733, 782]
    #
    # >>> token_boxes <- []
    # >>> for word, box in zip(words, normalized_word_boxes){
    # ...     word_tokens <- tokenizer$tokenize(word)
    # ...     token_boxes$extend([box] * len(word_tokens))
    # >>> # add bounding boxes of cls + sep tokens
    # >>> token_boxes <- [[0, 0, 0, 0]] + token_boxes + [[1000, 1000, 1000, 1000]]
    #
    # >>> encoding <- tokenizer(" ".join(words), return_tensors="pt")
    # >>> input_ids <- encoding$input_ids
    # >>> attention_mask <- encoding$attention_mask
    # >>> token_type_ids <- encoding$token_type_ids
    # >>> bbox <- torch::torch_tensor([token_boxes])
    #
    # >>> labels <- tokenizer("Hello world", return_tensors="pt")$input_ids
    #
    # >>> outputs <- model(
    # ...     input_ids=input_ids,
    # ...     bbox=bbox,
    # ...     attention_mask=attention_mask,
    # ...     token_type_ids=token_type_ids,
    # ...     labels=labels,
    # ... )
    #
    # >>> loss <- outputs$loss
    # ```"""
    return_dict <- append(return_dict, self$config$use_return_dict)[[1]]
    
    outputs <- self$layoutlm(
      input_ids,
      bbox,
      attention_mask=attention_mask,
      token_type_ids=token_type_ids,
      position_ids=position_ids,
      head_mask=head_mask,
      inputs_embeds=inputs_embeds,
      encoder_hidden_states=encoder_hidden_states,
      encoder_attention_mask=encoder_attention_mask,
      output_attentions=output_attentions,
      output_hidden_states=output_hidden_states,
      return_dict=return_dict,
    )
    
    sequence_output <- outputs[1]
    prediction_scores <- self$cls(sequence_output)
    
    masked_lm_loss <- NULL
    if (!is.null(labels)){
      loss_fct <- torch::nn_cross_entropy_loss()
      masked_lm_loss <- loss_fct(
        prediction_scores$view(-1, self$config$vocab_size),
        labels$view(-1),
      )
    }
    
    if (!return_dict){
      output <- append(prediction_scores, outputs[-c(1:2)])
      return(append(masked_lm_loss,output))
    }
    
    result <- list(
      loss=masked_lm_loss,
      logits=prediction_scores,
      hidden_states=outputs$hidden_states,
      attentions=outputs$attentions
    )
    class(result) <-  "MaskedLMOutput"
    return(result)
  }
)

#' LayoutLM Model with a sequence classification head on top (a linear layer on top of the pooled output) e.g. for
#' document image classification tasks such as the [RVL-CDIP](https://www.cs.cmu.edu/~aharley/rvl-cdip/) dataset.
LayoutLMForSequenceClassification<- torch::nn_module(
  "LayoutLMPreTrainedModel",
  initialize = function(config){
    self$num_labels <- config$num_labels
    self$layoutlm <- LayoutLMModel(config)
    self$dropout <- torch::nn_dropout(config$hidden_dropout_prob)
    self$classifier <- torch::nn_linear(config$hidden_size, config$num_labels)
    
    # Initialize weights and apply final processing
    # self$init_weights()
    
  },
  get_input_embeddings = function(){
    self$layoutlm$embeddings$word_embeddings
    
  },
  forward = function(
    input_ids=NULL,
    bbox=NULL,
    attention_mask=NULL,
    token_type_ids=NULL,
    position_ids=NULL,
    head_mask=NULL,
    inputs_embeds=NULL,
    labels=NULL,
    output_attentions=NULL,
    output_hidden_states=NULL,
    return_dict=NULL
  ){ # -> Unionc(Tuple, SequenceClassifierOutput):
    # r"""
    # labels (`torch::torch_LongTensor` of shape `(batch_size,)`, *optional*){
    #     Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
    #     config$num_labels - 1]`. If `config$num_labels == 1` a regression loss is computed (Mean-Square loss), If
    #     `config$num_labels > 1` a classification loss is computed (Cross-Entropy).
    #
    # Returns:
    #
    # Examples:
    #
    # ```python
    # >>> from transformers import LayoutLMTokenizer, LayoutLMForSequenceClassification
    # >>> import torch
    #
    # >>> tokenizer <- LayoutLMTokenizer$from_pretrained("microsoft/layoutlm-base-uncased")
    # >>> model <- LayoutLMForSequenceClassification$from_pretrained("microsoft/layoutlm-base-uncased")
    #
    # >>> words <- c("Hello", "world")
    # >>> normalized_word_boxes <- [637, 773, 693, 782], [698, 773, 733, 782]
    #
    # >>> token_boxes <- []
    # >>> for word, box in zip(words, normalized_word_boxes){
    # ...     word_tokens <- tokenizer$tokenize(word)
    # ...     token_boxes$extend([box] * len(word_tokens))
    # >>> # add bounding boxes of cls + sep tokens
    # >>> token_boxes <- [[0, 0, 0, 0]] + token_boxes + [[1000, 1000, 1000, 1000]]
    #
    # >>> encoding <- tokenizer(" ".join(words), return_tensors="pt")
    # >>> input_ids <- encoding$input_ids
    # >>> attention_mask <- encoding$attention_mask
    # >>> token_type_ids <- encoding$token_type_ids
    # >>> bbox <- torch::torch_tensor([token_boxes])
    # >>> sequence_label <- torch::torch_tensor([1])
    #
    # >>> outputs <- model(
    # ...     input_ids=input_ids,
    # ...     bbox=bbox,
    # ...     attention_mask=attention_mask,
    # ...     token_type_ids=token_type_ids,
    # ...     labels=sequence_label,
    # ... )
    #
    # >>> loss <- outputs$loss
    # >>> logits <- outputs$logits
    # ```"""
    return_dict <- append(return_dict,self$config$use_return_dict)[[1]]
    
    outputs <- self$layoutlm(
      input_ids=input_ids,
      bbox=bbox,
      attention_mask=attention_mask,
      token_type_ids=token_type_ids,
      position_ids=position_ids,
      head_mask=head_mask,
      inputs_embeds=inputs_embeds,
      output_attentions=output_attentions,
      output_hidden_states=output_hidden_states,
      return_dict=return_dict,
    )
    
    pooled_output <- outputs[1]
    
    pooled_output <- self$dropout(pooled_output)
    logits <- self$classifier(pooled_output)
    
    loss <- NULL
    if (!is.null(labels)){
      if (is.null(self$config$problem_type)){
        if (self$num_labels == 1){
          self$config$problem_type <- "regression"
        } else if (self$num_labels > 1 & (labels$dtype == torch::torch_long() | labels$dtype == torch::torch_int())){
          self$config$problem_type <- "single_label_classification"
        } else {
          self$config$problem_type <- "multi_label_classification"
        }
      }
      if (self$config$problem_type == "regression"){
        loss_fct <- torch::nn_mse_loss()
        if (self$num_labels == 1){
          loss <- loss_fct(logits$squeeze(), labels$squeeze())
        } else {
          loss <- loss_fct(logits, labels)
        }
      } else if (self$config$problem_type == "single_label_classification"){
        loss_fct <- torch::nn_cross_entropy_loss()
        loss <- loss_fct(logits$view(-1, self$num_labels), labels$view(-1))
      } else if (self$config$problem_type == "multi_label_classification"){
        loss_fct <- torch::nn_bce_with_logits_loss()
        loss <- loss_fct(logits, labels)
      }
    }
    if (!return_dict){
      output <- append(logits,outputs[-c(1:2)])
      return(append(loss, output)[[1]])
    }
    
    result <- list(
      loss=loss,
      logits=logits,
      hidden_states=outputs$hidden_states,
      attentions=outputs$attentions
    )
    class(result) <-  "SequenceClassifierOutput"
    return(result)
  }
)

#' LayoutLM Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for
#' sequence labeling (information extraction) tasks such as the [FUNSD](https://guillaumejaume.github.io/FUNSD/)
#' dataset and the [SROIE](https://rrc.cvc.uab.es/?ch=13) dataset.
LayoutLMForTokenClassification<- torch::nn_module(
  "LayoutLMPreTrainedModel",
  initialize = function(config){
    self$num_labels <- config$num_labels
    self$layoutlm <- LayoutLMModel(config)
    self$dropout <- torch::nn_dropout(config$hidden_dropout_prob)
    self$classifier <- torch::nn_linear(config$hidden_size, config$num_labels)
    
    # Initialize weights and apply final processing
    # self$init_weights()
    
  },
  get_input_embeddings = function(){
    self$layoutlm$embeddings$word_embeddings
    
  },
  forward = function(
    input_ids=NULL,
    bbox=NULL,
    attention_mask=NULL,
    token_type_ids=NULL,
    position_ids=NULL,
    head_mask=NULL,
    inputs_embeds=NULL,
    labels=NULL,
    output_attentions=NULL,
    output_hidden_states=NULL,
    return_dict=NULL
  ){   # -> Unionc(Tuple, TokenClassifierOutput):
    # r"""
    # labels (`torch::torch_LongTensor` of shape `(batch_size, sequence_length)`, *optional*){
    #     Labels for computing the token classification loss. Indices should be in `[0, ..., config$num_labels - 1]`.
    #
    # Returns:
    #
    # Examples:
    #
    # ```python
    # >>> from transformers import LayoutLMTokenizer, LayoutLMForTokenClassification
    # >>> import torch
    #
    # >>> tokenizer <- LayoutLMTokenizer$from_pretrained("microsoft/layoutlm-base-uncased")
    # >>> model <- LayoutLMForTokenClassification$from_pretrained("microsoft/layoutlm-base-uncased")
    #
    # >>> words <- c("Hello", "world")
    # >>> normalized_word_boxes <- [637, 773, 693, 782], [698, 773, 733, 782]
    #
    # >>> token_boxes <- []
    # >>> for word, box in zip(words, normalized_word_boxes){
    # ...     word_tokens <- tokenizer$tokenize(word)
    # ...     token_boxes$extend([box] * len(word_tokens))
    # >>> # add bounding boxes of cls + sep tokens
    # >>> token_boxes <- [[0, 0, 0, 0]] + token_boxes + [[1000, 1000, 1000, 1000]]
    #
    # >>> encoding <- tokenizer(" ".join(words), return_tensors="pt")
    # >>> input_ids <- encoding$input_ids
    # >>> attention_mask <- encoding$attention_mask
    # >>> token_type_ids <- encoding$token_type_ids
    # >>> bbox <- torch::torch_tensor([token_boxes])
    # >>> token_labels <- torch::torch_tensor([1, 1, 0, 0])$unsqueeze(0)  # batch size of 1
    #
    # >>> outputs <- model(
    # ...     input_ids=input_ids,
    # ...     bbox=bbox,
    # ...     attention_mask=attention_mask,
    # ...     token_type_ids=token_type_ids,
    # ...     labels=token_labels,
    # ... )
    #
    # >>> loss <- outputs$loss
    # >>> logits <- outputs$logits
    # ```"""
    return_dict <- append(return_dict,self$config$use_return_dict)[[1]]
    
    outputs <- self$layoutlm(
      input_ids=input_ids,
      bbox=bbox,
      attention_mask=attention_mask,
      token_type_ids=token_type_ids,
      position_ids=position_ids,
      head_mask=head_mask,
      inputs_embeds=inputs_embeds,
      output_attentions=output_attentions,
      output_hidden_states=output_hidden_states,
      return_dict=return_dict,
    )
    
    sequence_output <- outputs[[1]]
    
    sequence_output <- self$dropout(sequence_output)
    logits <- self$classifier(sequence_output)
    
    loss <- NULL
    if (!is.null(labels)){
      loss_fct <- torch::nn_cross_entropy_loss()
      loss <- loss_fct(logits$view(-1, self$num_labels), labels$view(-1))
    }
    
    if (!return_dict){
      output <- append(logits,outputs[-c(1:2)])[[1]]
      return(append(loss, output)[[1]])
    }
    
    result <- list(
      loss=loss,
      logits=logits,
      hidden_states=outputs$hidden_states,
      attentions=outputs$attentions
    )
    class(result) <-  "TokenClassifierOutput"
    return(result)
  },
  from_pretrained = function(pretrained_model_name) {
    .load_weights(pretrained_model_name)
  }
)

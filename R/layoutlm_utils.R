# few selected functions imported from transformers$modeling_utils$py

apply_chunking_to_forward <-  function(
    forward_fn, chunk_size, chunk_dim, input_tensors
){
#     """
#     This function chunks the `input_tensors` into smaller input tensor parts of size `chunk_size` over the dimension
#     `chunk_dim`. It then applies a layer `forward_fn` to each chunk independently to save memory.
#
#     If the `forward_fn` is independent across the `chunk_dim` this function will yield the same result as directly
#     applying `forward_fn` to `input_tensors`.
#
#     Args:
#         forward_fn (`Callable[..., torch::torch_Tensor]`){
#             The forward function of the model.
#         chunk_size (`int`){
#             The chunk size of a chunked tensor: `num_chunks <- len(input_tensors[0]) / chunk_size`.
#         chunk_dim (`int`){
#             The dimension over which the `input_tensors` should be chunked.
#         input_tensors (`Tuple[torch::torch_Tensor]`){
#             The input tensors of `forward_fn` which will be chunked
#
#     Returns:
#         `torch::torch_Tensor`: A tensor with the same shape as the `forward_fn` would have given if applied`.
#
#
#     Examples:
#
#     ```python
#     # rename the usual forward() fn to forward_chunk()
# }
# forward_chunk = function(self, hidden_states){
#         hidden_states <- self$decoder(hidden_states)
#         return hidden_states
#
#
#     # implement a chunked forward function
# }
# forward = function(self, hidden_states){
#         return apply_chunking_to_forward(self$forward_chunk, self$chunk_size_lm_head, self$seq_len_dim, hidden_states)
#     ```"""

    stopifnot("list(input_tensors) has to be a list of tensors" = len(input_tensors) > 0)

    # inspect$signature exist since python 3.5 and is a python method -> no problem with backward compatibility
    num_args_in_forward_chunk_fn <- len(inspect$signature(forward_fn)$parameters)
    # if (num_args_in_forward_chunk_fn != len(input_tensors)){
    #     raise ValueError(
    #         f"forward_chunk_fn expects list(num_args_in_forward_chunk_fn} arguments, but only {len(input_tensors)) input "
    #         "tensors are given"
    #     )
    # }
    #
    if (chunk_size > 0){
    #     tensor_shape <- input_tensors[0]$shape[chunk_dim]
    #     for (input_tensor in input_tensors){
    #         if (input_tensor$shape[chunk_dim] != tensor_shape){
    #             raise ValueError(
    #                 f"All input tenors have to be of the same shape: list(tensor_shape), "
    #                 f"found shape list(input_tensor$shape[chunk_dim])"
    #             )
    #         }
    #     }
    #
    #     if (input_tensors[0]$shape[chunk_dim] % chunk_size != 0){
    #         raise ValueError(
    #             f"The dimension to be chunked list(input_tensors[0]$shape[chunk_dim]) has to be a multiple of the chunk "
    #             f"size list(chunk_size)"
    #         )
    #     }
    #
        num_chunks <- input_tensors[0]$shape[chunk_dim] %/% chunk_size

        # chunk input tensor into tuples and apply forward fn to every tuple
        output_chunks <- purrr::map(input_tensors, ~forward_fn(.x$chunk(num_chunks, dim=chunk_dim)))
        # concatenate output at same dimension
        result <- torch::torch_cat(output_chunks, dim=chunk_dim)
    } else {
      result <- forward_fn(input_tensors)
    }
    result

}
find_pruneable_heads_and_indices <-  function(    heads, n_heads, head_size, already_pruned_heads){ # -> Tuplec(Set[int], torch::torch_LongTensor):
    # """
    # Finds the heads and their indices taking `already_pruned_heads` into account.
    #
    # Args:
    #     heads (`List[int]`): List of the indices of heads to prune.
    #     n_heads (`int`): The number of heads in the model.
    #     head_size (`int`): The size of each head.
    #     already_pruned_heads (`Set[int]`): A set of already pruned heads.
    #
    # Returns:
    #     `Tuplec(Set[int], torch::torch_LongTensor)`: A tuple with the remaining heads and their corresponding indices.
    # """
    mask <- torch::torch_ones(n_heads, head_size)
    heads <- set(heads) - already_pruned_heads  # Convert to set and remove already pruned heads
    for (head in heads){
        # Compute how many pruned heads are before the head and move the index accordingly
        head <- head - sum(already_pruned_heads < head)
        mask[head] <- 0
    }
    mask <- mask$view(-1)$contiguous()$eq(1)
    index <- torch::torch_arange(len(mask))[mask]$long()
    list(heads, index)

}
prune_linear_layer <-  function(layer, index, dim=0L){   # -> torch::nn_linear:
    # """
    # Prune a linear layer to keep only entries in index.
    #
    # Used to remove heads.
    #
    # Args:
    #     layer (`torch::torch_torch::nn_linear`): The layer to prune.
    #     index (`torch::torch_LongTensor`): The indices to keep in the layer.
    #     dim (`int`, *optional*, defaults to 0): The dimension on which to keep the indices.
    #
    # Returns:
    #     `torch::torch_torch::nn_linear`: The pruned layer as a new layer with `requires_grad=True`.
    # """
    index <- index$to(layer$weight$device)
    W <- layer$weight$index_select(dim, index)$clone()$detach()
    if (!is.null(layer$bias)){
        if (dim == 1){
            b <- layer$bias$clone()$detach()
        } else {
            b <- layer$bias[index]$clone()$detach()
        }
    }
    new_size <- list(layer$weight$size())
    new_size[dim] <- len(index)
    new_layer <- torch::nn_linear(new_size[1], new_size[0], bias=!is.null(layer$bias))$to(layer$weight$device)
    new_layer$weight$requires_grad <- FALSE
    new_layer$weight$copy_(W$contiguous())
    new_layer$weight$requires_grad <- TRUE
    if (!is.null(layer$bias)){
        new_layer$bias$requires_grad <- FALSE
        new_layer$bias$copy_(b$contiguous())
        new_layer$bias$requires_grad <- TRUE
    }
    new_layer
}

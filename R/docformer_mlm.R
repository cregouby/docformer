
# vdsr reconstruction as a port of https://github.com/twtygqyy/pytorch-vdsr/blob/master/vdsr.py
ltr_conv_relu_block <- torch::nn_module(
  "ltr_conv_relu_block",
  initialize = function() {
    self$conv <- torch::nn_conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, stride = 1, padding = 1, bias = FALSE)
    torch::nn_init_normal_(self$conv$weight, 0, sqrt(2 / (3 * 3 * 64)))
    self$relu <- torch::nn_relu(inplace = TRUE)

  },
  forward = function(x) {
    self$relu(self$conv(x))
  }
)

# from ResNet VAE Decoder https://github.com/hsinyilin19/ResNetVAE/blob/master/modules.py
ltr_head <- torch::nn_module(
  "ltr_head",
  initialize = function(config) {
    # CNN architechtures
    self$k1 <- c(5, 5)      # 2d kernal size
    self$k2 <- self$k3 <- self$k4 <- c(3, 3)      # 2d kernal size
    self$s1 <-  self$s2 <- self$s3 <- self$s4 <- c(2, 2)      # 2d strides
    self$pd1 <- self$pd2 <- self$pd3 <-  self$pd4 <- c(0, 0)  # 2d padding

    # Sampling vector :
    # TODO a nn_linear layer from x$shape[3] to 768 would be meaningfull here
    # Decoder
    # TODO two first sequential layers will be meaningless in case of low config$hidden_size
    stopifnot("ltr head is lacking support of hidden_size != 768" = config$hidden_size == 768 )
    self$convTrans6 <- torch::nn_sequential(
      torch::nn_conv_transpose2d(in_channels = 768, out_channels = 192, kernel_size = self$k4,
                                 stride = self$s4, padding = self$pd4),
      torch::nn_batch_norm2d(192, momentum = 0.01),
      torch::nn_relu(inplace = TRUE),
    )
    self$convTrans7 <- torch::nn_sequential(
      torch::nn_conv_transpose2d(in_channels = 192, out_channels = 48, kernel_size = self$k3,
                                stride = self$s3, padding = self$pd3),
      torch::nn_batch_norm2d(48, momentum = 0.01),
      torch::nn_relu(inplace = TRUE),
    )
    self$convTrans8 <- torch::nn_sequential(
      torch::nn_conv_transpose2d(in_channels = 48, out_channels = 12, kernel_size = self$k2,
                                stride = self$s2, padding = self$pd2),
      torch::nn_batch_norm2d(12, momentum = 0.01),
      torch::nn_relu(inplace = TRUE),
    )

    self$convTrans9 <- torch::nn_sequential(
      torch::nn_conv_transpose2d(in_channels = 12, out_channels = 3, kernel_size = self$k1,
                                stride = self$s1, padding = self$pd1),
      torch::nn_batch_norm2d(3, momentum = 0.01),
      torch::nn_sigmoid()    # y <- (y1, y2, y3) \in [0 ,1]^3
    )

    self$relu <- torch::nn_relu(inplace = TRUE)

  },
  forward = function(x) {
    x <- x$permute(c(1, 3, 2)) # "b s e -> b e s", batch, embedding, sequence
    img_reconstruct <- x$reshape(c(x$shape[1:2], 16, 32 )) # "b e s -> b e (wl.hl)", batch, embedding,  width_low, height_low, wl*hl=512
    img_reconstruct <-  self$convTrans6(img_reconstruct)
    img_reconstruct <-  self$convTrans7(img_reconstruct)
    img_reconstruct <-  self$convTrans8(img_reconstruct)
    img_reconstruct <-  self$convTrans9(img_reconstruct)
    img_reconstruct
  }
)

tdi_head <- torch::nn_module(
  "tdi_head",
  initialize = function(config) {
    self$tdi_classifier <- torch::nn_sequential(
      torch::nn_linear(config$hidden_size, 1),
      torch::nn_sigmoid()
    )
  },
  forward = function(x) {
    # classify "does_text_describe_image" from CLS embedding
    self$tdi_classifier(x[,1,])
  }
)

docformer_for_masked_lm <- torch::nn_module(
  "docformer_for_masked_LM",
  initialize = function(config, mask_id) {
    self$config <- config
    self$mask_id <- mask_id
    self$docformer <- docformer(config)

    self$mm_mlm <- LayoutLMLMPredictionHead(config)
    self$mlm_sigmoid <- torch::nn_sigmoid()
    self$ltr <- ltr_head(config)
    self$tdi <- tdi_head(config)

    self$mlm_loss <- torch::nn_cross_entropy_loss(ignore_index = mask_id)
    self$ltr_loss <- torch::nn_smooth_l1_loss()
    self$tdi_loss <- torch::nn_bce_with_logits_loss()
  },
  forward = function(x) {
    # compute sequence embedding
    embedding <- self$docformer(x)
    # compute Multi-Modal Masked Language Modeling (MM-MLM) and loss
    masked_x <- mask_for_tdi(mask_for_mm_mlm(x, self$mask_id))
    masked_embedding <- self$docformer(masked_x)
    # compute masked sequence embedding loss
    mm_mlm_loss <- self$mlm_loss(
      (self$mm_mlm(masked_embedding) %>% self$mlm_sigmoid())$movedim(1,2),
      (x$text + 1L)$squeeze(3)$to(torch::torch_long())
    )
    #  compute Learn To Reconstruct (LTR) the image and loss on images not masked by TDI
    ltr <- self$ltr(embedding)
    ltr_loss <- self$ltr_loss(
      torch::nnf_interpolate(ltr, x$image$shape[3:4]) * torch_tensor(!masked_x$image_mask)$reshape(-1,1,1,1),
      x$image * torch_tensor(!masked_x$image_mask)$reshape(-1,1,1,1)
    )
    # compute Text Describes Image (TDI) loss
    tdi <- self$tdi(masked_embedding)
    # compute loss
    masked_lm_loss <- (
      5 * mm_mlm_loss +
      ltr_loss +
      5 * self$tdi_loss(tdi, masked_x$image_mask)
    )

    # TODO extract other piggyback values see layoutlm_network.R @856
    result <- list(
      loss = masked_lm_loss,
      # hidden_states = embedding$hidden_states,
      # attentions = embedding$attentions
    )
    class(result) <- "MaskedLMOutput"
    return(result)
  }
)

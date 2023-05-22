#' Docformer Self-supervised training module
#'
#' @param ... Additional parameters passed to [docformer_config()].
#'
#' @describeIn docformer Pretraining module
#' @returns
#' A `luz_module` that has been setup and is ready to be `fitted`.
#'
#' @seealso [fit.luz_module_generator()] for fit arguments. See
#'   [predict.docformer_result()] for information on how to generate predictions.
#'
#' @export
docformer_pretrain <- function(..., mask_id = 0) {

  config <- docformer_config(...)

  module <- docformer_for_masked_lm %>%
    luz::setup(
      optimizer = optim_adam
      # metrics = list(
      #   self$mlm_loss,
      #   self$ltr_loss,
      #   self$tdi_loss
      # )
    ) %>%
    luz::set_hparams(
      config = config,
      mask_id = mask_id
    )

  attr(module, "module")$spec <- spec
  class(module) <- c("docformer_module", class(module))
  module
}

#' Fit the Temporal Fusion Transformer module
#'
#' @param object a TFT module created with [temporal_fusion_transformer()].
#' @param ... Arguments passed to [luz::fit.luz_module_generator()].
#'
#' @export
fit.docformer_module <- function(object, ...) {
  out <- NextMethod()
  class(out) <- c("docformer_result", class(out))
  out$spec <- attr(object, "module")$spec #preserve the spec in the result.

  # serialize the model, so saveRDS also works
  out$.serialized <- model_to_raw(out)

  out
}

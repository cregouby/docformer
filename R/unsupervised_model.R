# inspired by ?? need rework
docformer_mlm <- torch::nn_module(
  "docformer_mlm",
  initialize = function(config, lr = 5e-5, mask_id) {
    self$save_hyperparameters()
    self$docformer <- docformer_for_masked_lm(config, mask_id)
  },
  forward = function(x) {
    self$docformer(x)
  },
  training_step = function(batch, batch_idx) {
    logits <-  self$forward(batch)
    criterion <- unsupervised_loss()
    loss  <-  criterion(logits$transpose(2, 1), batch["mlm_labels"]$long())
    self$log("train_loss", loss, prog_bar = TRUE)

  },
  validation_step = function(batch, batch_idx) {
    logits  <-  self$forward(batch)
    criterion_mm_mlm <- torch$nn$CrossEntropyLoss()
    # TODO
    criterion_ltr <- torch$nn$SmoothedL1Loss()
    # TODO
    criterion_tdi <- torch$nn$CrossEntropyLoss()
    loss <- criterion_mm_mlm(logits$transpose(2, 1), batch["mlm_labels"]$long())
    val_acc <- 100 * (torch$argmax(logits, dim = -1) == batch["mlm_labels"]$long())$float()$sum() / (logits$shape[1] * logits$shape[2])
    val_acc <- torch$tensor(val_acc)
    self$log("val_loss", loss, prog_bar = TRUE)
    self$log("val_acc", val_acc, prog_bar = TRUE)
  },
  configure_optinizer = function() {
    torch::optim_adam(self$parameters(), lr = self$hparams["lr"])
  }
)

#' Docformer Self-supervised training task
#'
#' @param config config
#' @param train_dataloader training dataloader to use
#' @param val_dataloader validation dataloader to use
#' @param device "cpu" or "cuda" default to "auto"
#' @param epochs number of epoch to train
#' @param path path
#' @param classes number of classes
#' @param lr learning-rate
#' @param weights weights
#'
#' @return a docformer
#' @export
#'
docformer_pretrain <- function(config, train_dataloader, val_dataloader, device, epochs, path, classes, lr = 5e-5, weights = weights) {

}

# random_mlm_obfuscator <- torch::nn_module(
#   "random_obfuscator",
#   initialize = function(pretraining_ratio) {
#
#     if (pretraining_ratio <= 0 || pretraining_ratio >= 1) {
#       pretraining_ratio <- 0.15
#     }
#
#     self$pretraining_ratio <- pretraining_ratio
#
#   },
#   forward = function(text, x_feature) {
#     # workaround while torch_bernoulli is not available in CUDA
#     ones <- torch::torch_ones_like(text, device="cpu")
#     obfuscated_x <- torch::torch_bernoulli(self$pretraining_ratio * ones)
#     masked_input <- torch::torch_mul(obfuscated_x, x_feature)$to("cpu") %>% as_array
#
#     list(masked_input, obfuscated_vars)
#
#   }
# )

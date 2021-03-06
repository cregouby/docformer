
docformer_mlm <- torch::nn_module(
  "docformer_mlm",
  initialize = function(config,num_classes,lr = 5e-5){
    self$save_hyperparameters()
    self$docformer = docformer_for_classification(config, num_classes)
  },
  forward = function(x) {
    self$docformer(x)
  },
  training_step = function(batch,batch_idx) {
    logits  <-  self$forward(batch)
    criterion  <-  torch::nn_cross_entropy_loss()
    loss  <-  criterion(logits$transpose(2,1), batch["mlm_labels"]$long())
    self$log("train_loss",loss,prog_bar = True)

  },
  validation_step = function(batch,batch_idx) {
    logits  <-  self$forward(batch)
    criterion <- torch$nn$CrossEntropyLoss()
    loss <- criterion(logits$transpose(2,1), batch["mlm_labels"]$long())
    val_acc <- 100*(torch$argmax(logits,dim = -1)==batch["mlm_labels"]$long())$float()$sum()/(logits$shape[1]*logits$shape[2])
    val_acc <- torch$tensor(val_acc)
    self$log("val_loss", loss, prog_bar=True)
    self$log("val_acc", val_acc, prog_bar=True)
  },
  configure_optinizer = function() {
    torch::optim_adam(self$parameters(), lr=self$hparams["lr"])
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
#' @return
#' @export
#'
docformer_pretrain <- function(config,train_dataloader,val_dataloader,device,epochs,path,classes,lr = 5e-5,weights=weights) {

}

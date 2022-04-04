base_url <- "https://storage.googleapis.com/torchtransformers-models/"
hf_url_prefix <- "https://huggingface.co/"
hf_url_suffix <- "/resolve/main/pytorch_model.bin"
# maybe later store as a tibble with more info, but named vector is ok for now.

weights_url_map <- c(
  "microsoft/layoutlm-base-uncased" = paste0(
    hf_url_prefix,
    "microsoft/layoutlm-base-uncased",
    hf_url_suffix
  ),
  "microsoft/layoutlm-base-cased" = paste0(
    hf_url_prefix,
    "microsoft/layoutlm-base-cased",
    hf_url_suffix
  ),
  "microsoft/layoutlm-large-uncased" = paste0(
    hf_url_prefix,
    "microsoft/layoutlm-large-uncased",
    hf_url_suffix
  ),
  "allenai/hvila-block-layoutlm-finetuned-grotoap2" = paste0(
    hf_url_prefix,
    "allenai/hvila-block-layoutlm-finetuned-grotoap2",
    hf_url_suffix
  ),
  "allenai/hvila-row-layoutlm-finetuned-docbank" = paste0(
    hf_url_prefix,
    "allenai/hvila-row-layoutlm-finetuned-docbank",
    hf_url_suffix
  ),
  "allenai/ivila-block-layoutlm-finetuned-docbank" = paste0(
    hf_url_prefix,
    "allenai/ivila-block-layoutlm-finetuned-docbank",
    hf_url_suffix
  ),
  "hf-internal-testing/tiny-layoutlm" = paste0(
    hf_url_prefix,
    "hf-internal-testing/tiny-layoutlm",
    hf_url_suffix
  )
)


# There are some hard-to-avoid differences between the variable names in
# models constructed using this package and the standard variable names used in
# the huggingface saved weights. Here are some renaming rules that will (almost
# always?) be applied. We modify the *loaded* weights to match the *package*
# weights.
# Also! different models within huggingface have slightly different conventions!
# The tiny, etc.  models use "weight" & "bias" rather than "gamma" & "beta".

variable_names_replacement_rules <- c(
  ".gamma" = ".weight",
  ".beta" = ".bias",
  "LayerNorm" = "layer_norm",
  "attention.output.dense" = "attention.self.out_proj",
  "bert." = ""
)


# May as well store the configuration info for known layoutlm models here...
# "intermediate size" is always 4x the embedding size for these models.
transformer_configs <- tibble::tribble(
  ~model_name, ~embedding_size, ~n_layer, ~n_head, ~max_tokens, ~vocab_size,
  "microsoft/layoutlm-base-uncased", 768L, 12L, 12L, 512L, 50265L,
  "microsoft/layoutlm-base-cased", 768L, 12L, 12L, 512L, 50265L,
  "microsoft/layoutlm-large-uncased", 1024L, 24L, 16L, 512L, 30522L,
  "allenai/hvila-block-layoutlm-finetuned-grotoap2", 768L, 12L, 12L, 512L,  30522L,
  "allenai/hvila-row-layoutlm-finetuned-docbank", 768L, 12L, 12L, 512L,  30522L,
  "allenai/ivila-block-layoutlm-finetuned-docbank", 768L, 12L, 12L, 512L, 30522L,
  "hf-internal-testing/tiny-layoutlm", 32L, 2L, 2L, 512L, 5000L

)


usethis::use_data(
  weights_url_map,
  variable_names_replacement_rules,
  transformer_configs,
  internal = FALSE,
  overwrite = TRUE
)

rm(
  base_url,
  weights_url_map,
  variable_names_replacement_rules,
  transformer_configs
)


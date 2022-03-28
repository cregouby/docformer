## code to prepare `model_url` dataset goes here
hf_url_prefix <- "https://huggingface.co/"
hf_url_suffix <- "/resolve/main/pytorch_model.bin"

# "intermediate size" is always 4x the embedding size for these models.
transformers_config <- tibble::tribble(
  ~model_name, ~embedding_size, ~n_layer, ~n_head, ~max_tokens, ~vocab_size, ~n_labels, ~url, ~flavor,
  "layoutlm-base-uncased", 768L, 12L, 12L, 512L, 30522L,             2L,"microsoft/layoutlm-base-uncased", "hf",
  "layoutlm-base-cased", 768L, 12L, 12L, 512L, 50265L,               2L,"microsoft/layoutlm-base-cased",   "hf",
  "layoutlm-large-uncased", 1024L, 24L, 16L, 512L, 30522L,           2L,"microsoft/layoutlm-large-uncased","hf",
  "microsoft/layoutlm-base-uncased", 768L, 12L, 12L, 512L, 30522L,   2L,"microsoft/layoutlm-base-uncased", "hf",
  "microsoft/layoutlm-base-cased", 768L, 12L, 12L, 512L, 50265L,     2L,"microsoft/layoutlm-base-cased",   "hf",
  "microsoft/layoutlm-large-uncased", 1024L, 24L, 16L, 512L, 30522L, 2L,"microsoft/layoutlm-large-uncased","hf",
  "allenai/hvila-row-layoutlm-finetuned-docbank", 768L, 12L, 12L, 512L, 30522L, 13L, "allenai/hvila-row-layoutlm-finetuned-docbank","hf",
  "mrm8488/layoutlm-finetuned-funsd", 768L, 12L, 12L, 512L, 30522L, 13L,"mrm8488/layoutlm-finetuned-funsd","hf",
  "clee7/layoutlm-finetune-sroie", 768L, 12L, 12L, 512L, 30522L, 13L,   "clee7/layoutlm-finetune-sroie",   "hf"
) %>%
  dplyr::mutate(url = dplyr::case_when(flavor=="hf" ~ paste0(hf_url_prefix,url,hf_url_suffix)))



usethis::use_data(
  transformers_config,
  internal = TRUE,
  overwrite = TRUE
)
rm(
  hf_url_prefix,
  hf_url_suffix,
  transformers_config
)

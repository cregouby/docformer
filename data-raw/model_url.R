## code to prepare `model_url` dataset goes here
hf_url_prefix <- "https://huggingface.co/"
hf_model_suffix <- "/resolve/main/pytorch_model.bin"
hf_tokenizer_suffix <- "/resolve/main/tokenizer.json"
gh_url_prefix <- "https://media.githubusercontent.com/media/cregouby/docformer_models/main/inst/"
gh_url_suffix <- ".pth"

# "intermediate size" is always 4x the embedding size for these models.
transformers_config <- tibble::tribble(
  ~model_name, ~hidden_size, ~n_layer, ~n_head, ~max_position_embeddings, ~max_2d_position_embeddings, ~vocab_size, ~intermediate_ff_size_factor, ~n_labels, ~url, ~flavor,
  "layoutlm-base-uncased", 768L, 12L, 12L, 512L, 1024L, 30522L, 4L,             2L,"microsoft/layoutlm-base-uncased", "gh",
  # "layoutlm-base-cased", 768L, 12L, 12L, 512L, 1024L, 50265L, 4L,               2L,"microsoft/layoutlm-base-cased",   "hf",
  # "layoutlm-large-uncased", 1024L, 24L, 16L, 512L, 1024L, 30522L,  4L,          2L,"microsoft/layoutlm-large-uncased","hf",
  "microsoft/layoutlm-base-uncased", 768L, 12L, 12L, 512L, 1024L, 30522L, 4L,   2L,"microsoft/layoutlm-base-uncased", "gh",
  "microsoft/layoutlm-base-cased", 768L, 12L, 12L, 512L, 1024L, 50265L, 4L,     2L,"microsoft/layoutlm-base-cased",   "hf",
  "microsoft/layoutlm-large-uncased", 1024L, 24L, 16L, 512L, 1024L, 30522L, 4L, 2L,"microsoft/layoutlm-large-uncased","hf",
  "allenai/hvila-row-layoutlm-finetuned-docbank", 768L, 12L, 12L, 512L, 1024L, 30522L, 4L   , 13L, "allenai/hvila-row-layoutlm-finetuned-docbank","gh",
  "allenai/hvila-block-layoutlm-finetuned-grotoap2", 768L, 12L, 12L, 512L, 1024L, 30522L, 4L, 13L, "allenai/hvila-block-layoutlm-finetuned-grotoap2","gh",
  # "mrm8488/layoutlm-finetuned-funsd", 768L, 12L, 12L, 512L, 1024L, 30522L, 4L,  13L,"mrm8488/layoutlm-finetuned-funsd","hf",
  # "clee7/layoutlm-finetune-sroie", 768L, 12L, 12L, 512L, 1024L, 30522L, 4L,     13L,"clee7/layoutlm-finetune-sroie",   "hf",
  "hf-internal-testing/tiny-layoutlm", 32L, 2L, 2L, 512L, 128L, 5000L, 2L, 2L, "hf-internal-testing/tiny-layoutlm", "gh"
) %>%
  dplyr::mutate(
    tokenizer_json = paste0(hf_url_prefix, url, hf_tokenizer_suffix),
    url = dplyr::case_when(flavor=="hf" ~ paste0(hf_url_prefix, url, hf_model_suffix),
                           flavor=="gh" ~ paste0(gh_url_prefix, stringr::str_split(url,"/", simplify=TRUE)[,2], gh_url_suffix))
  )



usethis::use_data(
  transformers_config,
  internal = FALSE,
  overwrite = TRUE
)
rm(
  hf_url_prefix,
  hf_url_suffix,
  gh_url_prefix,
  gh_url_suffix,
  transformers_config
)

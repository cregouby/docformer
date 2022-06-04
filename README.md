
<!-- README.md is generated from README.Rmd. Please edit that file -->

# docformer

<!-- badges: start -->

[![Lifecycle:
experimental](https://img.shields.io/badge/lifecycle-experimental-orange.svg)](https://lifecycle.r-lib.org/articles/stages.html)
<!-- badges: end -->

{docformer} is an implementation of [DocFormer: End-to-End Transformer
for Document Understanding](https://arxiv.org/abs/2106.11539) relying on
[torch for R](https://torch.mlverse.org/resources/) providing a
multi-modal transformer based architecture for the task of Visual
Document Understanding (VDU) 📄📄📄, as a port of
[shabie/docformer](https://github.com/shabie/docformer) code.

DocFormer uses text, vision and spatial features and combines them using
a novel multi-modal self-attention layer. DocFormer can be pre-trained
in an unsupervised fashion using carefully designed tasks which
encourage multi-modal interaction. DocFormer also shares learned spatial
embeddings across modalities which makes it easy for the model to
correlate text to visual tokens and vice versa. DocFormer is evaluated
on 4 different datasets each with strong baselines. DocFormer achieves
state-of-the-art results on all of them, sometimes beating models 4x
larger in no. of parameters. ![High-level Neural Netword design with
building-blocks around the Docformer Multi-Modal transformer
](man/figure/Simplistic_design.jpg)

## Installation

You can install the development version of docformer like so:

``` r
# install.packages("remotes")
remotes::install_github("cregouby/docformer")
```

docformer currently supports the `{sentencepiece}` package for
tokenization prerequisites, and relies on `{pdftools}` for
digitally-born pdfs, and `{tesseract}` with `{magick}` for OCR
documents.

``` r
if (! ("sentencepiece" %in% rownames(installed.packages()))) { install.packages("sentencepiece") }
```

## Usage Example

![Side-by-side document ground truth and docformer prediction with
superimposed color: red for title, blue for question, green for answer
](man/figure/README_result.jpg)

This is a basic workflow to train a docformer model:

### Turn a document into input tensor

``` r
library(sentencepiece)
library(docformer)
# get the corpus
doc <- pins::pin("https://arxiv.org/pdf/2106.11539.pdf")

# load a sentencepiece tokenizer and add a <mask> token if needed
tok_model <- sentencepiece::sentencepiece_load_model(system.file(package="sentencepiece", "models/nl-fr-dekamer.model"))
tok_model$vocab_size <- tok_model$vocab_size+1L
tok_model$vocabulary <- rbind(tok_model$vocabulary, data.frame(id=tok_model$vocab_size, subword="<mask>"))

# turn the document into docformer input tensor
doc_tensor <- create_features_from_doc(doc = doc, tokenizer = tok_model)
```

### Import a pretrained model

``` r
config  <-  docformer_config(pretrained_model_name = "microsoft/layoutlm-base-uncased")
docformer_model <- docformer(config)
```

### or shape your own model

``` r
config  <-  docformer_config(hidden_size = 76L, max_position_embeddings = 52L, num_attention_heads = 4L,num_hidden_layers = 3L, vocab_size = 5000L, device = "cpu")
docformer_model <- docformer(config)
```

### Pretrain the model (Work in progress)

A self-supervised training task can be run with

``` r
# train a model from that tensor
# docformer_ssl <- docformer_pretrain(doc_tensor, epochs=30)
```

### Train the model (work in progress)

…followed by a supervised training task on some annotated documents…

``` r
# docformer_model <- docformer_fit(doc_tensor, from_model=docformer_ssl, epochs=30)
```

### Predict with the model

Predict with the headless model gives a document-layout embedding tensor
of shape \[ <document pages>, <max_position_embeddings>, <hidden_size>
\]

``` r
doc_embedding <- docformer_model(doc_tensor)
```

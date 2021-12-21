
<!-- README.md is generated from README.Rmd. Please edit that file -->

# docformer

<!-- badges: start -->
<!-- badges: end -->

{docformer} is an implementation of [DocFormer: End-to-End Transformer
for Document Understanding](https://arxiv.org/abs/2106.11539) relying on
[torch fro R](https://torch.mlverse.org/resources/) providing a
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
larger in no. of parameters.

## Installation

You can install the development version of docformer like so:

``` r
# install.packages("remotes")
remotes::install_github("mlverse/docformer")
```

## Example

![Side-by-side document ground truth and docformer prediction with
superimposed color: red for title, blue for question, green for answer
](man/figure/README_result.jpg)

This is a basic example which shows you how to solve a common problem:

``` r
#library(docformer)
## basic example code
```

% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/prepare_dataset.R
\name{create_features_from_doc}
\alias{create_features_from_doc}
\title{Turn document into docformer torch tensor input feature}
\usage{
create_features_from_doc(
  doc,
  tokenizer,
  add_batch_dim = TRUE,
  target_geometry = "384x500",
  max_seq_len = 512,
  apply_mask_for_mlm = FALSE,
  extras_for_debugging = FALSE
)
}
\arguments{
\item{doc}{file path, url, or raw vector to document (currently pdf only)}

\item{tokenizer}{tokenizer function to apply to words extracted from image. Currently,
{hftokenizers}, {tokenizer.bpe} and {sentencepiece} tokenizer are supported.}

\item{add_batch_dim}{(boolean) add a extra dimension to tensor for batch encoding}

\item{target_geometry}{image target magik geometry expected by the image model input}

\item{max_seq_len}{size of the embedding vector in tokens}

\item{apply_mask_for_mlm}{add mask to the language model}

\item{extras_for_debugging}{additionnal feature for debugging purposes}

\item{save_to_disk}{(boolean) shall we save the result onto disk}

\item{path_to_save}{result path}
}
\value{
a list of named tensors
}
\description{
Turn document into docformer torch tensor input feature
}
\examples{
# load a tokenizer with <mask> encoding capability
sent_tok <- sentencepiece::sentencepiece_load_model(
   system.file(package="sentencepiece", "models/nl-fr-dekamer.model")
   )
sent_tok$vocab_size <- sent_tok$vocab_size+1L
sent_tok$vocabulary <- rbind(
  sent_tok$vocabulary,
  data.frame(id=sent_tok$vocab_size, subword="<mask>")
  )
# turn pdf into feature
doc <- system.file(package="docformer", "inst", "2106.11539_1_2.pdf")
doc_tt <- create_features_from_doc(doc, tokenizer=sent_tok)

}

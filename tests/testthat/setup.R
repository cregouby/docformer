# Run before any test
sent_tok <- sentencepiece::sentencepiece_load_model(system.file(package="sentencepiece", "models/nl-fr-dekamer.model"))
# prepend tokenizer with missing tokens
sent_tok_mask <- sent_tok
sent_tok_mask$vocab_size <- sent_tok_mask$vocab_size+2L
sent_tok_mask$vocabulary <- rbind(data.frame(subword=c("<mask>","<pad>")),sent_tok_mask$vocabulary["subword"]) %>% tibble::rowid_to_column("id") %>% dplyr::mutate(id=id-1)

bpe_tok <- tokenizers.bpe::bpe_load_model(system.file(package="tokenizers.bpe", "extdata/youtokentome.bpe"))
bpe_tok_mask <- bpe_tok
bpe_tok_mask$vocab_size <- bpe_tok_mask$vocab_size+2L
bpe_tok_mask$vocabulary <- rbind( data.frame(subword=c("<MASK>","<PAD>")),bpe_tok_mask$vocabulary["subword"]) %>% tibble::rowid_to_column("id") %>% dplyr::mutate(id=id-1)

# hf_tok <- hftokenizers::(system.file(package="sentencepiece", "models/nl-fr-dekamer.model"))

image <- system.file(package="docformer", "inst", "2106.11539_1.png")
doc <- system.file(package="docformer", "inst", "2106.11539_1_2.pdf")
doc2 <- system.file(package="docformer", "inst", "OSI_1_2.pdf")

doc_tt <- create_features_from_doc(doc, sent_tok_mask)
tiny_tt <- create_features_from_doc(doc, sent_tok_mask, target_geometry = "128x128")
tiny_tt$text <- tiny_tt$text$clamp_max(5000)

# Run after all tests
withr::defer(testthat::teardown_env())

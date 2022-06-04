# Run before any test
sent_tok <- sentencepiece::sentencepiece_load_model(system.file(package="sentencepiece", "models/nl-fr-dekamer.model"))
sent_tok_mask <- sent_tok
sent_tok_mask$vocab_size <- sent_tok_mask$vocab_size+1L
sent_tok_mask$vocabulary <- rbind(sent_tok_mask$vocabulary, data.frame(id=sent_tok_mask$vocab_size, subword="<mask>"))

bpe_tok <- tokenizers.bpe::bpe_load_model(system.file(package="tokenizers.bpe", "extdata/youtokentome.bpe"))
bpe_tok_mask <- bpe_tok
bpe_tok_mask$vocab_size <- bpe_tok_mask$vocab_size+1L
bpe_tok_mask$vocabulary <- rbind(bpe_tok_mask$vocabulary, data.frame(id=bpe_tok_mask$vocab_size, subword="<MASK>"))

# hf_tok <- hftokenizers::(system.file(package="sentencepiece", "models/nl-fr-dekamer.model"))

image <- system.file(package="docformer", "inst", "2106.11539_1.png")
doc <- system.file(package="docformer", "inst", "2106.11539_1_2.pdf")
doc2 <- system.file(package="docformer", "inst", "OSI_1_2.pdf")

doc_tt <- create_features_from_doc(doc, sent_tok_mask)

# Run after all tests
withr::defer(testthat::teardown_env())

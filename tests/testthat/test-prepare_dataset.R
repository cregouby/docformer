test_that("normalize_box works with single var", {
  expect_equal(normalize_box(c(22,34,27,41), width=100, height=100, size=100), c(22,34,27,41))
})

# test_that("normalize_box works with dataframe", {
#   df <- tibble::tibble(bbox=c(list(22,34,27,41),list(22,34,27,41)),  width=c(100,100), height=c(100,100), size=c(100,100))
#   expect_equal(all(normalize_box(df), c(22,34,27,41)))
# })

test_that(".tokenize return a flat list", {
  vocab <- "Cooking, cookery, or culinary arts is the art, science, and craft of using heat to prepare food for consumption. Cooking techniques and ingredients vary widely, from grilling food over an open fire to using electric stoves, to baking in various types of ovens, reflecting local conditions.
Types of cooking also depend on the skill levels and training of the cooks. Cooking is done both by people in their own dwellings and by professional cooks and chefs in restaurants and other food establishments.
Preparing food with heat or fire is an activity unique to humans. It may have started around 2 million years ago, though archaeological evidence for the same does not predate more than 1 million years.
The expansion of agriculture, commerce, trade, and transportation between civilizations in different regions offered cooks many new ingredients. New inventions and technologies, such as the invention of pottery for holding and boiling of water, expanded cooking techniques. Some modern cooks apply advanced scientific techniques to food preparation to further enhance the flavor of the dish served."
  sent_tok <- c("This", "Simple", "Coconut", "Curry", "Recipe", "Produces", "Flavorful", "Fish", "Fast")
  tmp <- tempfile()
  readr::write_lines(vocab, file=tmp)
  sentpiece <- sentencepiece::sentencepiece(tmp, type="bpe", vocab_size = 10000, verbose=F )

  expect_true(purrr::vec_depth(.tokenize(tokenizer=sentpiece, sent_tok))==1)
  })


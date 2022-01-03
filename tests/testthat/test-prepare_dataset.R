test_that("normalize_box works with single var", {
  expect_equal(normalize_box(c(22,34,27,41), width=100, height=100, size=100), c(22,34,27,41))
})

test_that("normalize_box works with dataframe", {
  df <- data.frame(bbox=c(c(22,34,27,41),c(22,34,27,41)),  width=c(100,100), height=c(100,100), size=c(100,100))
  expect_equal(all(normalize_box(df), c(22,34,27,41)))
})

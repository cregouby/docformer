Sys.setenv(KMP_DUPLICATE_LIB_OK = TRUE)
# torch_zeros(1, names="hello") # trigger warning about named tensors

skip_if_not_test_examples <- function() {
  if (Sys.getenv("TEST_EXAMPLES", unset = "0") != "1") {
    skip("Not testing examples/readme. Set the env var TEST_EXAMPLES = 1.")
  }
}

skip_if_cuda_not_available <- function() {
  if (!cuda_is_available()) {
    skip("A GPU is not available for testing.")
  }
}

torch_mem_used <- function() {
  all_mem <- lobstr:::new_bytes(
    system("pmap $(pidof rsession) | tail -n1", intern = TRUE) %>%
    stringr::str_extract("\\d+") %>%
    as.numeric * 1000
    )
  r_mem <- lobstr::mem_used() %>% as.numeric
  all_mem - r_mem
}

torch_obj_size <- function(obj) {
  stopifnot("object is not a torch tensor" = inherits(obj, "torch_tensor"))
  mem_before <- torch_mem_used()
  add_obj_mem <- torch::torch_reshape(obj, obj$shape)
  mem_after <- torch_mem_used()
  mem_after - mem_before
}

expect_equal_to_tensor <- function(object, expected, ...) {
  expect_equal(as.array(object), as.array(expected), ...)
}

expect_not_equal_to_tensor <- function(object, expected) {
  expect_false(isTRUE(all.equal(as.array(object), as.array(expected))))
}

expect_no_error <- function(object, ...) {
  expect_error(object, NA, ...)
}

expect_tensor <- function(object) {
  expect_true(torch:::is_torch_tensor(object))
  # workaround torch_Half dtype not handled by as.array
  if (as.character(object$dtype) == "Half") {
    expect_no_error(as.array(object$to(dtype = torch::torch_float())$cpu()))
  } else {
    expect_no_error(as.array(object$cpu()))
  }
}

expect_equal_to_r <- function(object, expected, ...) {
  # workaround torch_Half dtype not handled by as.array
  if (as.character(object$dtype) == "Half") {
    expect_equal(as.array(object$to(dtype = torch::torch_float())$cpu()), expected, ...)
  } else {
    expect_equal(as.array(object$cpu()), expected, ...)
  }

}

expect_tensor_shape <- function(object, expected) {
  expect_tensor(object)
  expect_equal(object$shape, expected)
}

expect_tensor_dtype <- function(object, expected_dtype) {
  expect_tensor(object)
  expect_equal(as.character(object$dtype), expected_dtype)
}

expect_undefined_tensor <- function(object) {
  # TODO
}

expect_identical_modules <- function(object, expected) {
  expect_identical(
    attr(object, "module"),
    attr(expected, "module")
  )
}

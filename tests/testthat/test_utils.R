test_that("torch_obj_size works with tensors of different dtypes", {
  expect_equal(torch_obj_size(torch::torch_ones(c(2,7,4))$to(dtype = torch::torch_bool())), lobstr:::new_bytes(56))
  expect_equal(torch_obj_size(torch::torch_ones(c(2,7,4))$to(dtype = torch::torch_int())), lobstr:::new_bytes(1792))
  expect_equal(torch_obj_size(torch::torch_ones(c(2,3,1))$to(dtype = torch::torch_long())), lobstr:::new_bytes(384))
  expect_equal(torch_obj_size(torch::torch_ones(c(2,3,1))$to(dtype = torch::torch_int32())), lobstr:::new_bytes(192))
  expect_equal(torch_obj_size(torch::torch_ones(c(2,3,1))$to(dtype = torch::torch_int16())), lobstr:::new_bytes(96))
  expect_equal(torch_obj_size(torch::torch_ones(c(2,3,4))$to(dtype = torch::torch_float())), lobstr:::new_bytes(768))
  expect_equal(torch_obj_size(torch::torch_ones(c(2,3,4))$to(dtype = torch::torch_float16())), lobstr:::new_bytes(384))
  expect_equal(torch_obj_size(torch::torch_ones(c(2,3,1))$to(dtype = torch::torch_float64())), lobstr:::new_bytes(384))
})

test_that("torch_obj_size works with nn_modules", {
  expect_equal(torch_obj_size(docformer_net), lobstr:::new_bytes(56))
})

test_that("torch_obj_size works with docformer_tensor", {
  expect_equal(torch_obj_size(doc_tt),
               list(x_features = lobstr:::new_bytes(196610), y_features = lobstr:::new_bytes(196610),
                    text = lobstr:::new_bytes(32770), image = lobstr:::new_bytes(9160000),
                    mask = lobstr:::new_bytes(1020)), tolerance = .1)
})


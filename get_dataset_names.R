rm(list = ls())
library(tidyverse)

split_names <- list.files("qsar", pattern = "*.csv")

gsub('_training_preprocessed.csv|_test_preprocessed.csv', '', split_names) %>%
  unique() %>%
  data.frame() %>%
  write_csv('qsar_datasets.txt', col_names = FALSE)



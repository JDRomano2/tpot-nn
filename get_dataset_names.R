rm(list = ls())
library(tidyverse)

split_names <- list.files("qsar", pattern = "*.csv") %>%
  strsplit(split = '_') %>%
  map(1) %>%
  unlist() %>%
  unique() %>%
  data.frame() %>%
  write_csv('qsar_datasets.txt', col_names = FALSE)


rm(list = ls())
library(tidyverse)

split_names <- here::here('code', 'qsar') %>% 
  list.files(pattern = "*.csv")

gsub('_training_preprocessed.csv|_test_preprocessed.csv', '', split_names) %>%
  unique() %>%
  data.frame() %>%
  write_csv(here::here('code', 'qsar_datasets.txt'), col_names = FALSE)

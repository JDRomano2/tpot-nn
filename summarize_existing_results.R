library(tidyverse)

dbn_desc <- read_csv('existing_performance/dbn_dataset_description.csv')
dbn <- read_csv('existing_performance/dbn_dnn.csv') %>%
  merge(dbn_desc, by = 'Dataset index')
dnn <- read_delim('existing_performance/dnn_results.txt', delim = ' ')

dbnn <- merge(dnn, dbn, by.x = 'dataset', by.y = 'Dataset') %>%
  mutate(ghasemi_dnn_r2 = round(DNN^2,3)) %>%
  select(dataset, ghasemi_dnn_r2, dnn_median, dnn_tuned, dnn_best, CD, PCD, FPCD, DNN) %>%
  rename(ma_dnn_tuned_r2 = dnn_tuned,
         ma_dnn_best_r2 = dnn_best,
         ma_dnn_median_r2 = dnn_median,
         ghasemi_cd_r = CD,
         ghasemi_pcd_r = PCD,
         ghasemi_fpcd_r = FPCD,
         ghasemi_dnn_r = DNN) %>%
  write_csv('existing_performance/dbnn.csv')

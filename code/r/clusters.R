library(cluster)    # clustering algorithms
library(factoextra) # clustering algorithms & visualization
library(tidyverse)
library(here)

# function to compute average silhouette for k clusters
avg_sil <- function(k, df) {
  km.res <- kmeans(df, centers = k, nstart = 25)
  ss <- silhouette(km.res$cluster, dist(df))
  mean(ss[, 3])
}

k.values <- 2:15

mysets <- here('code', 'qsar_datasets.txt') %>% 
  read_csv(col_names = F) %>% 
  pull(X1)

# Nfiles = 15
for (set in mysets ) {
  print(Nfiles)
  trainFile <- paste0('qsar/', set, '_training_preprocessed.csv')

  # using colClasses to speed up reading of files
  train <- read.csv(trainFile, header=TRUE, nrows=100)
  classes <- sapply(train,class)
  train <- read.csv(trainFile, header=TRUE, colClasses=classes)
  
  df <- train %>%  dplyr::select(-c(MOLECULE, Act)) %>% t()
  avg_sil_values <- map_dbl(k.values, avg_sil, df)
  
  best_k <- 
  
  k_clus <- kmeans(t(df), centers = best_k, nstart = 25)
  
  grouping <- data.frame(Subset = k_clus$cluster, feature = colnames(df))
  subset <- grouping %>%
    group_by(Subset) %>%
    summarise(Features = paste0(feature, collapse = ";")) %>%
    mutate(SubsetSize = count(grouping, Subset)$n)
  
  write_csv(subset, file = here('code', 'subsets', paste0('sub', set, '.csv')))
  
}

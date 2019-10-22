library(cluster)    # clustering algorithms
library(factoextra) # clustering algorithms & visualization
library(tidyverse)
# trainFile <- paste0('qsar/', split_names[10])
# train <- read.csv(trainFile, header=TRUE, nrows=100)
# # train <- read.csv(trainFile, header=TRUE, nrows=100)
# boxplot(train[, 3:100])
# vars <- apply(train[, 3:100], 2, var)
# plot(vars)
# str(train)
# classes <- sapply(train, class);
# train <- read.csv(trainFile, header=TRUE, colClasses=classes)
# 
# train2 <- read.csv(paste0('qsar/', split_names[12]), header=TRUE)
# # train <- read.csv(trainFile, header=TRUE, nrows=100)
# # boxplot(train2[, 3:100])
# vars <- apply(train2[, 3:100], 2, var)
# summary(vars)
# # str(train2)                

# function to compute average silhouette for k clusters
avg_sil <- function(k, df) {
  km.res <- kmeans(df, centers = k, nstart = 25)
  ss <- silhouette(km.res$cluster, dist(df))
  mean(ss[, 3])
}

k.values <- 2:15

mysets <- read.csv('qsar_datasets.txt', header = F)[,1]
set = mysets[1]
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
  
  write_csv(subset, file = paste0('subsets/sub', set, '.csv'))
  
}

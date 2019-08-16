library(cluster)    # clustering algorithms
library(factoextra) # clustering algorithms & visualization
library(tidyverse)
        
# function to compute average silhouette for k clusters
.get_ave_sil_width <- function(d, cluster){
  ss <- cluster::silhouette(cluster, d)
  mean(ss[, 3])
}

mysets <- read.csv('qsar_datasets.txt', header = F, stringsAsFactors = F)[, 1]
k.max <- 20

for (set in mysets) {
  print(set)
  trainFile <- paste0('qsar/', set, '_training_preprocessed.csv')
  
  # using colClasses to speed up reading of files
  train <- read.csv(trainFile, header=TRUE, nrows=100)
  classes <- sapply(train,class)
  train <- read.csv(trainFile, header=TRUE, colClasses=classes)
  df <- train %>%  dplyr::select(-c(MOLECULE, Act)) %>% t()
  
  diss <- wordspace::dist.matrix(as.matrix(df), method = 'euclidean', as.dist = TRUE)
  v <- rep(0, k.max)
  for (i in 2:k.max) {
    print(i)
    clust <- kmeans(as.matrix(df), i)
    v[i] <- .get_ave_sil_width(diss, clust$cluster)
  }

  best_k <- which.max(v)
  k_clus <- kmeans(df, centers = best_k, nstart = 25)
  grouping <- data.frame(Subset = k_clus$cluster, feature = rownames(df)) 
  subset <- grouping %>%
    group_by(Subset) %>%
    summarise(Features = paste0(feature, collapse = ";")) %>%
    mutate(SubsetSize = count(grouping, Subset)$n)
  
  write_csv(subset, paste0('subsets/sub', set, '.csv'))
}

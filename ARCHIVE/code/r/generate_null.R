rm(list = ls())
n.preds <- 10
null.fac <- 5 # number of (factored) NULL columns, 5 = five times as many
set.seed(12345)

n.preds <- 10
data.dir <- "data_300"
filenames <- list.files(data.dir, pattern="*.txt", full.names=TRUE)
filenames.short <- gsub(pattern = paste0(data.dir, "/"), replacement = "", filenames)
filenames.short <- gsub(pattern = ".txt", replacement = "", filenames.short)

old.cols <- paste0("X", 0:(n.preds-1))
new.cols <- paste0("X", (n.preds):(null.fac*n.preds-1))
write.table(old.cols, paste0("subset_", null.fac, "/functional_vars.csv"), row.names = F, col.names = F)
perm.new.cols <- sample(new.cols, length(new.cols), replace = F)


for (j in 1:(null.fac-1)){
  write.table(perm.new.cols[((j-1)*n.preds+1):(j*n.preds)], 
              paste0("subset_", null.fac, "/noisy_vars_", j, ".csv"), 
              row.names = F, col.names = F)
}


for (i in 1:length(filenames.short)){
  my.dat <- read.csv(paste0(data.dir, "/", filenames.short[i], ".txt"), sep = "\t")
  ori.dat <- my.dat[, !(colnames(my.dat) %in% c("Distribution", "Fitness"))]
  n.samp <- nrow(my.dat)

  my.dat[, new.cols] <- matrix(sample(0:2, n.preds*n.samp, replace = T), ncol = n.preds, nrow = n.samp, byrow = T)
  my.dat <- my.dat[, c(old.cols, new.cols, "Class")]
  write.csv(my.dat, file = paste0("csv_", data.dir, "/with_null_", null.fac, "_nullfac_", filenames.short[i] , ".csv"), 
            row.names = F)
}





# my.dat <- read.csv("datasets/AAA_bench_mod_4_up_down_dt_gb_score_0.35173_75.txt", sep = "\t")

my.dat <- read.csv("datasets/AAA_bench_mod_4_up_down_gb_dt_score_0.16366_5.txt", sep = "\t")
n.preds <- 10
n.samp <- nrow(my.dat)
old.cols <- paste0("X", 0:(n.preds-1))
new.cols <- paste0("X", (n.preds):(2*n.preds-1))
my.dat[, new.cols] <- matrix(sample(0:2, n.preds*n.samp, replace = T), ncol = n.preds, nrow = n.samp, byrow = T)
my.dat <- my.dat[, c(old.cols, new.cols, "Class")]
write.csv(my.dat, file = "datasets/with_null_AAA_bench_mod_4_up_down_gb_dt_score_0.16366_5.csv", 
          row.names = F)
write.csv(old.cols, "functional_vars.csv", row.names = F)
write.csv(new.cols, "noisy_vars.csv", row.names = F)

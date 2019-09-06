setwd("~/Google Drive/Study/Data Science Masters/Modules/Foundations of DS/Coursework/CW1")
fd <- read.table ("fish.txt", header = F)
colnames(fd) <- c("Time", "Weight")
times <- fd$Time
minutes <- times - floor(times)
weights <- fd$Weight
hist(times, breaks = (0:24), probability=T)
lines( density ( times ), col="red" )
hist(weights, breaks = (0:20)/4, probability=T)
lines( density ( weights ), col="red" )
plot(times ~ weights)
plot(minutes ~ weights)
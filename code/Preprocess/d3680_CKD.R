########################################################
# All rights reserved. 
# Author: XIE Zhengwei @ Beijing Gigaceuticals Tech Co., Ltd 
#                      @ Peking University International Cancer Institute
# Contact: xiezhengwei@gmail.com
#
#
########################################################

#######################################################
#To calculate the CDK figureprint of molecules

library(rcdk)
dat1 <- read.csv("used_drug_3680_smi.csv",header=FALSE,blank.lines.skip=FALSE)
#smi <-lapply(as.character(dat1$V1),parse.smiles)
cmp.fp<-vector("list",nrow(dat1))
print(nrow(dat1))
for (i in 1:nrow(dat1)){
  smi <- lapply(as.character(dat1[i,]),parse.smiles)
  if (!is.null(unlist(smi[[1]]))) {
  cmp.fp[i] <- lapply(smi[[1]][1],get.fingerprint,type="circular")
  }
  print(i)
}

# For each mol, calculate CDK fingerprint
for (i in 1:nrow(dat1)){
  print(i)
  if(is.null(cmp.fp[[i]])) {
    a_mat <- matrix(nrow=1,ncol=1024)
        for (j in 1:ncol(a_mat)) {
            a_mat[j] <- 0
        }
  } else {
    a_mat <- fingerprint::fp.to.matrix(cmp.fp[i])
  }
  if(i>1) {
    b_mat <- rbind(b_mat,a_mat)
  } else {
    b_mat <- a_mat
  }
}

#Save to file
dim(b_mat)
write.table(b_mat,file="d3680_CDK_fingureprints.txt")     


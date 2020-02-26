#LOADING LIBRARIES
library(MASS)
library(ggplot2)
library(tidyr)
library(tidyverse)
#REQUIRED FUNCTIONS
pval <- function(x){2-2*pnorm(abs(x))}

Bonferroni <- function(X,s,alfa,n){
  TruePositive <- mean(abs(X[1:s]) > qnorm(1- alfa/(2*n)))
  FalsePositive <- (sum(abs(X[(s+1):n]) > qnorm(1-alfa/(2*n))) >= 1)
  return(data.frame(TruePositive,FalsePositive))
}

Sidak <- function(X,s,alfa,n){
  TruePositive <- mean(abs(X[1:s]) > qnorm(0.5*(1+(1-alfa)^(1/n))))
  FalsePositive <- (sum(abs(X[(s+1):n]) > qnorm(0.5*(1+(1-alfa)^(1/n)))) >= 1)
  return(data.frame(TruePositive,FalsePositive))
}

Smax <- function(X,s,alfa,n,theta,CovMatrix){
  Y <- apply(t(apply(mvrnorm(1000,rep(0,nrow(CovMatrix)),CovMatrix),1,abs)),1,max)
  TruePositive <- mean(abs(X[1:s]) > unname(quantile(Y,1-alfa)))
  FalsePositive <- (sum(abs(X[(s+1):n]) > unname(quantile(Y,1-alfa))) >= 1)
  return(data.frame(TruePositive,FalsePositive))
}

Nk_0 <- function(X,alfa){
  k = 1
  PVals <- sort(pval(X))
  n = length(X)
  #print(PVals)
  #print(PVals[k]<=alfa/(length(X)+k-1))
  while(PVals[k]<=(alfa/(length(X)+k-1))){
    k=k+1
    if(k==length(X)){
      return(k)
    }
  }
  N0 <- 1:n
  Nk <- N0 #searching Nk0 for current X
  Nk1 <- N0[PVals >= alfa/length(Nk)]
  while(length(Nk) != length(Nk1)) {
    Nk <- Nk1
    Nk1 <- N0[PVals >= alfa/length(Nk)]
  }
  return(k)
}

Holm <- function(X,s,alfa,n){
  k <- Nk_0(X,alfa)
  TruePositive <- mean(abs(X[1:s]) > qnorm(1 - alfa/(2*(n-k+1))))
  FalsePositive <- (sum(abs(X[(s+1):n]) > qnorm(1-alfa/(2*(n-k+1)))) >= 1)
  return(data.frame(TruePositive,FalsePositive))
}

Sk_0 <- function(X,alfa){
  k = length(X)
  PVals <- sort(pval(X))
  while(PVals[k]>alfa/(n-k+1)){
    k=k-1
    if(k==1){
      return(k)
    }
  }
  return(k)
}

Hochberg <- function(X,s,alfa,n){
  k <- Sk_0(X,alfa)
  TruePositive <- mean(abs(X[1:s]) > qnorm(1- alfa/(2*(n-k+1))))
  FalsePositive <- (sum(abs(X[(s+1):n]) > qnorm(1-alfa/(2*(n-k+1)))) >= 1)
  return(data.frame(TruePositive,FalsePositive))
}

#MAIN 

alfa <- 0.05
n <- 10
Correlations <- c(0,0.5,0.9)
No.NonNullComps <- c(1,3,8)
PowerInterval <- seq(0,6,0.1)

# Correlations <- 0
# No.NonNullComps <- 1
# PowerInterval <- seq(0,6,0.1)
Procedures <- c("Bonferroni","Sidak","Holm","Hochberg","Smax")
#Procedures <- c("Smax")
# theta <- c(rep(2,1),rep(0,n-1))
# CovMatrix <- matrix(c(1,rep(0,n)),n,n,byrow = T)
# 
#par(mfrow=c(2,2))

for(s in No.NonNullComps){
  for(ro in Correlations){
    PowerDF <- as.data.frame(matrix(0,length(PowerInterval),5))
    FWERDF <- as.data.frame(matrix(0,length(PowerInterval),5))
    for(Procedure in Procedures){
      Power <- c()
      FWER <- c()
      for(c in PowerInterval){
        theta <- c(rep(c,s),rep(0,n-s))
        CovMatrix <- matrix(c(1,rep(ro,n)),n,n,byrow = T)
        TruePositives <- c()
        FalsePositives <- c()
        if(Procedure == "Smax"){
          for(i in 1:200){
            X <- mvrnorm(1,theta,CovMatrix)
            TruePositives[i] <- get(Procedure)(X,s,alfa,n,theta,CovMatrix)$TruePositive
            FalsePositives[i] <- get(Procedure)(X,s,alfa,n,theta,CovMatrix)$FalsePositive
          }
        } else {
          for(i in 1:200){
            X <- mvrnorm(1,theta,CovMatrix)
            TruePositives[i] <- get(Procedure)(X,s,alfa,n)$TruePositive
            FalsePositives[i] <- get(Procedure)(X,s,alfa,n)$FalsePositive
          }
        }
        Power[match(c,PowerInterval)] <- mean(TruePositives)
        FWER[match(c,PowerInterval)] <- mean(FalsePositives)
      }
      PowerDF[,match(Procedure,Procedures)] = Power
      FWERDF[,match(Procedure,Procedures)] = FWER
      # PowerPlot <- PowerPlot + geom_line(aes(x=seq(0,6,0.1), y = Power),col = match(Procedure,Procedures))
      # FWERPlot <- FWERPlot + geom_line(aes(x=seq(0,6,0.1), y = FWER),col = match(Procedure,Procedures))
    }
    colnames(PowerDF) <- Procedures
    colnames(FWERDF) <- Procedures
    PowerDF <- cbind(Theta = PowerInterval, PowerDF)
    PowerDF <- gather(PowerDF, 'method','Power',-Theta)
    assign(paste("PowerDF","s=",s,'ro=',ro),PowerDF)
    PowerPlot <- ggplot(PowerDF, aes(Theta,y = Power, color = method)) + geom_line() + ggtitle(paste("s= ",s,"ro= ",ro))
    ggsave(file = paste("PowerPlot","s=",s,"ro=",ro,".pdf",sep = "_"),plot = PowerPlot,path=getwd())
    FWERDF <- cbind(Theta = PowerInterval, FWERDF)
    FWERDF <- gather(FWERDF, 'method','FWER',-Theta)
    assign(paste("FWERDF","s=",s,'ro=',ro),FWERDF)
    FWERPlot <- ggplot(FWERDF, aes(Theta,y = FWER, color = method)) + geom_line() + ggtitle(paste("s= ",s,"ro= ",ro))
    ggsave(file = paste("FWERPlot","s=",s,"ro=",ro,".pdf",sep = "_"),plot = FWERPlot,path = getwd())
  }
}


#dev.off()
# library(tidyr)
# df <- cbind(x = seq(0,6,0.1), PowerDF)
# df <- gather(df, 'method','value',-x)
# ggplot(df, aes(x,y = value, color = method)) + geom_line()
# for(const in consts){
#   TruePositives <- c()
#   FalsePositives <- c()
#   theta <- c(rep(const,s),rep(0,n-s))
#   for(j in 1:100){
#     X <- mvrnorm(1,theta,CovMatr)
#     
#     Nk_0 <- 0
#     TruePositives[j] <- pval(X[1:s]) < alfa/Nk_0
#     FalsePositives[j] <- pval(X[s:n]) < alfa/Nk_0
#   }
#   AvgPower[match(const,consts)] <- mean(TruePositives)
#   FWER[match(const,consts)] <- mean(FalsePositives)
# }

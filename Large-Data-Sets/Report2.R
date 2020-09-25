#LOADING LIBRARIES
library(MASS)
library(ggplot2)
library(tidyr)
library(tidyverse)


l_max <- function(x,alpha){
  n <- 10
  l <- 1:10
  p_value_sorted <- sort(pval(x))
  p_value_idx_sorted <- order(pval(x))
  rejected_ratio <- (alpha*l)/n
  rejection <- p_value_sorted < rejected_ratio
  l_max <- max(which(rejection))
}




Benjamin_Hochberg <- function(x,s){
  l_mx <-l_max(x,0.05)
  wrong_rejection <-sum( pval(x)[(s+1):10] < (l_mx*alpha/10) ) 
  FDP <- (wrong_rejection/max(sum(pval(x)<l_mx*alpha/10),1))
  FWER <- wrong_rejection >=  1 
  POWER <- sum(pval(x)[1:s]<(l_mx*alpha/10))/s  
  results<- data.frame(FDP ,FWER ,POWER )
  return(results)
}


corrMatrix <- rep

alfa <- 0.05 
n <- 10 

# EXERCISE 1
Correlations <- c(0)
# EXERCISE 2
Correlations <- c(0,0.3,0.6,0.9)

No.NonNullComps <- c(3,5,8)
PowerInterval <- seq(0,6,0.25)
No.VectorsToGenerate <- 10000

Correlations <- 0
No.NonNullComps <- 1
PowerInterval <- seq(0,6,0.3)
theta <- c(rep(2,1),rep(0,n-1))
CovMatrix <- matrix(c(1,rep(0,n)),n,n,byrow = T)



list_of_list <- list()
for(s in No.NonNullComps ){
  j = 1
  for(ro in Correlations){
    i = 1
    list_constant_s_diff_corr <- list()
    BenjaminiHochbergDF <- data.frame(cbind(PowerInterval,matrix(0,length(PowerInterval),3)))
    colnames(BenjaminiHochbergDF) <- c("Theta","FDR","FWER","Power")
    for (Theta in PowerInterval){
      cat("jestem w ", Theta)
      theta <- c(rep(Theta,times = s),rep(0,times = n-s))
      CovMatrix <- matrix(c(1,rep(ro,n)),n,n) 
      X <- mvrnorm(No.VectorsToGenerate,theta,CovMatrix)
      ProcedureBH_result <- apply(X,MARGIN = 1,Benjamin_Hochberg,s = s)
      Procedure_DF<-bind_rows(ProcedureBH_result)
      Procedure_DF_final <- apply(Procedure_DF,MARGIN = 2,mean) 
      BenjaminiHochbergDF[match(Theta,PowerInterval),]<- c(Theta,Procedure_DF_final)
    }
    list_constant_s_diff_corr[[i]] <-  BenjaminiHochbergDF
    i = i+1
  }
  list_of_list[[j]] <- list_constant_s_diff_corr
  }


library(ggplot2)

ggplot(BenjaminiHochbergDF,aes(x = Theta))+
  geom_line(aes(y = Power),col = 'red')

ggplot(BenjaminiHochbergDF,aes(x = Theta))+
  geom_line(aes(y = FDR),col = 'red')+
  geom_line(aes(y = FWER),col = 'blue')


library("reshape2")

test_data_long <- melt(BenjaminiHochbergDF[,1:3], id="Theta")


ggplot(data =test_data_long,aes(x = Theta,y = value,colour = variable) ) + 
  geom_line()

get_corr_matrix <- function(ro,n = 10) {
  # matrix(c(1,rep(ro,n)),n,n) 
  matrix(ro,nrow = n,ncol = n) + diag(1-ro,n,n)
}

get_corr_matrix(0.3)

get_single_experiment <- function(s,ro,c) {
  multi_vectors <- get_design_matrix(s,ro,c)
  ProcedureBH_result <- apply(multi_vectors,MARGIN = 1,Benjamin_Hochberg,s = s)
  Procedure_DF <- bind_rows(ProcedureBH_result)
  Procedure_DF_final <- lapply(Procedure_DF,mean)
  cbind(ro = ro,s=s,c=c,as.data.frame(Procedure_DF_final))
  }

get_single_experiment(5,0.3,3)

get_design_matrix <- function(s,ro,c) {
  mvrnorm(1000,c(rep(c,times = s),rep(0,times = 10-s)),get_corr_matrix(ro))
}

results <- lapply(c(3, 5, 8), function(n_non_zero) {
  lapply(c(0, 0.3, 0.6, 0.9), function(ro) {
    lapply(seq(0,6,by = 0.25),function(c) {
      get_single_experiment(n_non_zero,ro,c)
    } )
  } )
} )

results_table <- bind_rows(
  unlist(
    unlist(
      results,recursive = FALSE,use.names = FALSE
    ),
    recursive = FALSE,use.names = FALSE
  )
)

library(tidyr)

results_table %>%
  gather('statistic', 'value', FDP, FWER, POWER) %>%
  ggplot(aes(x = c, y = value, color = as.factor(as.character(ro)))) +
  geom_line() +
  facet_grid(statistic~s) +
  theme_bw()

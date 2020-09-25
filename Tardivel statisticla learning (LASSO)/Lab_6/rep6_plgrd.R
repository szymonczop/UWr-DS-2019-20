library(MASS)
#library(ADMM)

X1 <-  mvrnorm(n=10, mu = c(0,0,0), Sigma = diag(3)) # W wierszu jedna zmiannX 

mean(rowSums((X1 - 1)**2)/dim(X1)[1])

n <- 10 

normal_mse <- c()
js_mse <- c()

place <- 1
set.seed(2020)
for(mu_val in seq(0,7,by = 0.01)){
  
  X_experiment <- mvrnorm(n = 10000,mu = rep(mu_val,times = 10),Sigma = diag(10))
  normal_mse[place] <- mean(rowSums((X_experiment - mu_val)^2)/dim(X_experiment)[2])
  JStein_matrix <- X_experiment - ((n-2)*X_experiment)/rowSums(X_experiment^2)
  js_mse[place] <- mean(rowSums((JStein_matrix - mu_val)^2)/dim(JStein_matrix)[2])
  
  place <- place + 1
}

plot(seq(0,7,by = 0.01),js_mse,type = "l",col = "red",xlab = "mu value",ylab = "MSE")
lines(seq(0,7,by = 0.01),normal_mse,type = "l",col = "blue")
legend(2, 0.6, legend=c("James Stein", "MSE Z"),
       col=c("red", "blue"), lty=1:2, cex=0.5)




############# ZJEBANY DOWÓD ####################

########## KOLEJNE ZADANKO #################

n <- 10
p <- 5
sigma <- 1
alpha <- 0.05 
beta <- c(3,1,0,0,0)

library(pracma)
set.seed(2020)
X = randortho(10)[, 1 : 5]
eps = rnorm(10)


Y <- X %*% beta + eps

beta_ols <- t(X) %*% Y


lasso_estim <- function(b_ols,lambda){
           beta_lasso <- c()
           place <- 1
           for (val in b_ols){
             beta_lasso[place] <- sign(val)*max(abs(val)-lambda,0)
             place <- place + 1
           }
           return (beta_lasso)
}

lasso_estim(beta_ols,lambda) 

lasso_loss <- function(beta_lasso,Y,X){
  first_part <-sum((Y- X %*% beta_lasso)^2)
  second_part <- sum(beta_lasso != 0) *2
  final <- first_part + second_part -5
  return(final)
}


lambda_minimizer <- function(b_ols,X,Y){
  
  scores <- c()
  place <- 1
  for(lambda in seq(0.01,4,by = 0.001)){
    
    LASSO_beta <- lasso_estim(b_ols,lambda) 
    loss <- lasso_loss(LASSO_beta,Y,X)
    #cat(loss,"\n")
    scores[place] <- loss
    place <- place + 1
  }
  
  return(scores)
  
}

func<- lambda_minimizer(beta_ols,X,Y)

plot( seq(0.01,4,by = 0.001),func,type = "l",col = "red",xlab = "lambda",ylab = "Lasso loss")
abline(v = abs(beta_ols)[1],col = "blue")
abline(v = abs(beta_ols)[2],col = "blue")
abline(v = abs(beta_ols)[3],col = "blue")
abline(v = abs(beta_ols)[4],col = "blue")
abline(v = abs(beta_ols)[5],col = "blue")



seq(0.01,4,by = 0.001)[which(min(func) == func)] # optymalna lambda

########################ZAD 3


beta.0 <- c()
beta.1 <- c()
place <- 1
for(experiment in 1:10000){
  
  cat(experiment,"\n")
  
  eps = rnorm(10)
  Y <- X %*% beta + eps
  beta_ols <- t(X) %*% Y
  
  func_val <- lambda_minimizer(beta_ols,X,Y)
  lambda1_optim <- seq(0.01,4,by = 0.001)[which(min(func_val) == func_val)]
  beta_lasso1 <- lasso_estim(beta_ols,lambda1_optim ) 
  
  lambda0_optmi <- qnorm((1 + (1 - alpha)^(1/p))/2)
  beta_lasso0 <- lasso_estim(beta_ols,lambda0_optmi)
  
  beta.0[place] <- sum( (X%*%beta_ols - X %*% beta_lasso0)^2 )
  beta.1[place] <- sum( (X%*%beta_ols - X %*% beta_lasso1)^2 )
  
  place <- place + 1
}

plot(1:10000,sort(beta.0),col = "green",type = 'l',ylab = "Error value")
lines(1:10000,sort(beta.1),col = "red",type = "l")
abline(h = mean(beta.0),col = "darkgreen")
abline(h = mean(beta.1),col = "darkred")
legend(2000, 25, legend=c("Lambda 0", "Lambda 1","mean beta0","mean beta1"),
       col=c("green", "red","darkgreen","darkred"), lty=1:2, cex=0.5)

c(mean(beta.0),mean(beta.1))


########### PODEJSĆIE Z 2 strony 


betas.0 <- c()
betas.1 <- c()
place <- 1
for(experiment in 1:10000){
  
  cat(experiment,"\n")
  
  eps = rnorm(10)
  Y <- X %*% beta + eps
  beta_ols <- t(X) %*% Y
  
  func_val <- lambda_minimizer(beta_ols,X,Y)
  lambda1_optim <- seq(0.01,4,by = 0.001)[which(min(func_val) == func_val)]
  beta_lasso1 <- lasso_estim(beta_ols,lambda1_optim ) 
  
  lambda0_optmi <- qnorm((1 + (1 - alpha)^(1/p))/2)
  beta_lasso0 <- lasso_estim(beta_ols,lambda0_optmi)
  
  betas.0[place] <- lasso_loss(beta_lasso0,Y,X)
  betas.1[place] <- lasso_loss(beta_lasso1,Y,X)
  
  place <- place + 1
}


plot(1:10000,sort(betas.0),col = "green",type = 'l',ylab = "Error value")
lines(1:10000,sort(betas.1),col = "red",type = "l")
abline(h = mean(betas.0),col = "darkgreen")
abline(h = mean(betas.1),col = "darkred")
legend(2000, 35, legend=c("Lambda 0", "Lambda 1","mean beta0","mean beta1"),
       col=c("green", "red","darkgreen","darkred"), lty=1:2, cex=0.5)



################# lST 



prob_beta0 <- c()
prob_beta1 <- c()
place <- 1

for(experiment in 1:10000){
  cat(experiment,"\n")
  
  eps = rnorm(10)
  Y <- X %*% beta + eps
  beta_ols <- t(X) %*% Y
  
  func_val <- lambda_minimizer(beta_ols,X,Y)
  lambda1_optim <- seq(0.01,4,by = 0.001)[which(min(func_val) == func_val)]
  beta_lasso1 <- lasso_estim(beta_ols,lambda1_optim ) 
  
  lambda0_optmi <- qnorm((1 + (1 - alpha)^(1/p))/2)
  beta_lasso0 <- lasso_estim(beta_ols,lambda0_optmi)
  
  if(beta_lasso0[3] != 0 || beta_lasso0[4] != 0 || beta_lasso0[5] != 0 ) {prob_beta0[place] <- FALSE} else {prob_beta0[place] <- TRUE} 
  if(beta_lasso1[3] != 0 || beta_lasso1[4] != 0 || beta_lasso1[5] != 0 ) {prob_beta1[place] <- FALSE} else {prob_beta1[place] <- TRUE} 
  
  
  place <- place + 1
}



1 - sum(prob_beta0)/length(prob_beta0)
1 - sum(prob_beta1)/length(prob_beta1)
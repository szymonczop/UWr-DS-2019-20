library(MASS)
covMat <- matrix(c(1,rep(0,100)),ncol = 100,nrow = 100)
theta <- c(1,1,1,1,1,rep(0,95))
x<- mvrnorm(n=1,mu = theta,covMat)
alpha <-  0.05
prob_t <- c()
stat <-c()
end_theta <- c()
for(j in 1:10){
  theta<- c(j,rep(0,99))
  prob_t <- c()
  stat <-c()
for(i in 1:1000){
  x<- mvrnorm(n=1,mu = theta,covMat)
  t <- max(abs(x))
  prob_t[i]<- t > qnorm(1-alpha/200)
  stat[i] <- sum(prob_t)/i
}
  end_theta[j] <- stat[i]
  }
  

#plot(stat,type = 'l')
plot(end_theta,type = 'l')
abline(v = sqrt(2*log(200)),col = 'red')


######## 2
covMat <- matrix(c(1,rep(0,100)),ncol = 100,nrow = 100)
theta <- c(rep(1,100))
prob_t <- c()
stat <-c()
end_power <- c()
p <- 1
for(j in seq(-2,2,by = 0.1)){
  theta<- c(rep(j,100))
  prob_t <- c()
  stat <-c()
  for(i in 1:100){
    x<- mvrnorm(n=1,mu = theta,covMat)
    t <- sum(x^2)
    prob_t[i]<- t > qchisq(0.95,100)
    stat[i] <- sum(prob_t)/i
                }
  end_power[p] <- stat[i]
  p <- p+1
  }

plot(seq(-2,2,by = 0.1),end_power, type= 'l')

#####B
prob_t_2 <- c()
p <- 1
for(j in seq(0,2,by = 0.1)){
  theta<- c(rep(j,100))
  x<- mvrnorm(n=1,mu = theta,covMat)
  mu_c <- sqrt(50)*j^2
  a <- qnorm(0.95) - mu_c
  b <- sqrt(1+mu_c*sqrt(8/100))
  prob_t_2[p] <-1- pnorm(a/b)
  p <- p+1
  }

plot(seq(0,2,by = 0.1),prob_t_2,type = 'l',col = 'red')
lines(seq(0,2,by = 0.1),end_power[20:40])


##### EXERCISE 3
n <- 1000
n2 <- 1:1000
dzielnik <-n / n2
wynik <- c()
for(i in 1:100000){ 
x <- runif(n,min = 0,max = 1)
dzielnik <-n / n2
wynik[i] <- min(sort(x)*dzielnik)
}

plot(density(wynik))

#EXERCISE4

covMat <- matrix(c(1,rep(0,100)),ncol = 100,nrow = 100)

#theta <- c(1,1,1,1,1,rep(0,95))
#x<- mvrnorm(n=1,mu = theta,covMat)
alpha <-  0.05
prob_t <- c()
stat <-c()
end_theta <- c()

idx1<- 1

par(mfrow = c(1,3))

n <- 100
n2 <- 1:100
dzielnik <-n / n2

for(s in c(1,5,20,100)){
  c <- s
  idx1 <- 1
  end_theta <-c()
  end_power<-c()
  p_value_mean<-c()
  
for(j in seq(-2,2,by = 0.1)){
  theta<- c(rep(j,c),rep(0,100-c))
  
  
  prob_t <- c()
  stat <-c()
  prob_t_chi<- c()
  stat_t_chi<-c()
  
  p_value <-c()
  
  
  for(i in 1:1000){
    x<- mvrnorm(n=1,mu = theta,covMat)
    t <- max(abs(x))
    prob_t[i]<- t > qnorm(1-alpha/200)
    stat[i] <- sum(prob_t)/i
    
    t_chi <- sum(x^2)
    prob_t_chi[i]<- t_chi > qchisq(0.95,100)
    stat_t_chi[i] <- sum(prob_t_chi)/i
    
    p_value[i]<- min(sort(2*(1-pnorm(abs(x))))*dzielnik) < 0.05
    
  }
  end_theta[idx1] <- stat[i]
  end_power[idx1] <- stat_t_chi[i]
  p_value_mean[idx1] <- sum(p_value)/1000
  idx1 <- idx1+1
  print(str(idx1))
}
  idx <- 1
  plot(seq(-2,2,by = 0.1),end_theta,type = 'l',col = 'green')
  plot(seq(-2,2,by = 0.1),end_power, type= 'l',col = 'red')
  plot(seq(-2,2,by = 0.1),p_value_mean,type = 'l',col = 'blue')
}










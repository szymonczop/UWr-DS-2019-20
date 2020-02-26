
n <- 100
c <- 6
theta <-c(rep(6,times =5 ),rep(0,times = 5))

covMat <- matrix(c(1,rep(0,100)),ncol = 100,nrow = 100)
x<- mvrnorm(n=1,mu = theta,covMat)
alpha <-  0.05

avg_power<-c()
true_reject <- c()
prob_reject <- c()

for(i in 1:n){
  theta <-c(rep(6,times =i ),rep(0,times = n-i))
  for(j in 1:100){
    x<- mvrnorm(n=1,mu = theta,covMat)
    true_reject[j] <-sum( abs(x[1:i]) > qnorm(1- alpha/(2*n))) / i 
  }

  avg_power[i] = sum(true_reject)/length(true_reject)
}

avg_power

###### ZAD2
library(ggplot2)
power_c <- c()
i = 1
s <- 1
kupa <- seq(0,6,by = 0.1)
for(k in seq(0,6,by = 0.1)){
  theta <- c(rep(k,times = s),rep(0,times = n-s))
  for(j in 1:100){
    x<- mvrnorm(n=1,mu = theta,covMat)
    true_reject[j] <-sum( abs(x[1:s]) > qnorm(1- alpha/(2*n))) / s
  }
  power_c[i] <- sum(true_reject)/length(true_reject)
  i = i+1
}



power_bonferroni<- qplot(kupa,power_c,main = 'Bonferroni')


##### ZAD 3 


i <- 1
c <- 1
n <- 10 
covMat <- matrix(c(1,rep(0,10)),ncol = 10,nrow = 10)
#false_reject <- c()
#par(mfrow = c(1,2))
FWER<- matrix(0,ncol = 3,nrow = 61)

for(s in c(1,5,8)){
  
  for(k in seq(0,6,by = 0.1)){
    
    theta <- c(rep(k,times = s),rep(0,times = n-s))
    
    for(j in 1:100){
      x<- mvrnorm(n=1,mu = theta,covMat)
      if(sum(abs(x[(s+1):length(x)]) > qnorm(1- alpha/(2*n))) ==0 ){
        false_reject[j] <- 0
      }
      else{
        false_reject[j] <- 1
      }
      
       
    }
    FWER[i,c] <- sum(false_reject)/length(false_reject)
    i = i+1
    print(str(i))
    
  }
  c <- c+1
  i <- 1
  #plot(seq(0,6,by = 0.1),FWER)
}


ggplot(as.data.frame(FWER),aes(x =seq(0,6,by = 0.1),y  = FWER[,1] ))+
  geom_line()+
  geom_hline(yintercept = mean(FWER[,1]))+
  geom_line(aes(y = FWER[,2]),col = 'red')+
    geom_hline(yintercept = mean(FWER[,2]),col = 'red')+
  geom_line(aes(y = FWER[,3]),col = 'blue')+
    geom_hline(yintercept = mean(FWER[,3]),col = 'blue')+
  ggtitle('Bonferroni FWER')


#### EXERCISE 2
covMat <- matrix(c(1,rep(0,100)),ncol = 100,nrow = 100)
power_c <- c()
i = 1
alpha <- 0.05
s <- 5 # to sobie sam ustalam 
kupa <- seq(0,6,by = 0.1)
for(k in seq(0,6,by = 0.1)){
  theta <- c(rep(k,times = s),rep(0,times = n-s))
  for(j in 1:100){
    x<- mvrnorm(n=1,mu = theta,covMat)
    true_reject[j] <-sum( x[1:s] > qnorm(0.5*(1+(1-alpha)^(1/100)))) / s
  }
  power_c[i] <- sum(true_reject)/length(true_reject)
  i = i+1
}


power_sidlak<-qplot(kupa,power_c,main = 'Sidlak')



require(gridExtra)

grid.arrange(power_bonferroni, power_sidlak, ncol=2)


###b)
i <- 1
c <- 1
n <- 10 
covMat <- matrix(c(1,rep(0,10)),ncol = 10,nrow = 10)
#false_reject <- c()
#par(mfrow = c(1,2))
FWER<- matrix(0,ncol = 3,nrow = 61)

for(s in c(1,5,8)){
  
  for(k in seq(0,6,by = 0.1)){
    
    theta <- c(rep(k,times = s),rep(0,times = n-s))
    
    for(j in 1:100){
      x<- mvrnorm(n=1,mu = theta,covMat)
      if(sum(abs(x[(s+1):length(x)]) > qnorm(0.5*(1+(1-alpha)^(1/10)))) ==0 ){
        false_reject[j] <- 0
      }
      else{
        false_reject[j] <- 1
      }
      
      
    }
    FWER[i,c] <- sum(false_reject)/length(false_reject)
    i = i+1
    print(str(i))
    
  }
  c <- c+1
  i <- 1
  #plot(seq(0,6,by = 0.1),FWER)
}




ggplot(as.data.frame(FWER),aes(x =seq(0,6,by = 0.1),y  = FWER[,1] ))+
  geom_line()+
  geom_hline(yintercept = mean(FWER[,1]))+
  geom_line(aes(y = FWER[,2]),col = 'red')+
  geom_hline(yintercept = mean(FWER[,2]),col = 'red')+
  geom_line(aes(y = FWER[,3]),col = 'blue')+
  geom_hline(yintercept = mean(FWER[,3]),col = 'blue')+
  ggtitle('Siak FWER')


#####c)




#####EXERCISE 3
p<- 0.5
n <- 10

#covMat <- matrix(c(1,rep(p,n)),ncol = n,nrow = n)
alpha <- 0.05
theta <- rep(0,times = n)
max <- c()
quantile <- c()
j <- 1

for(p in c(0,0.5,0.9,1)){
  covMat <- matrix(c(1,rep(p,n)),ncol = n,nrow = n)
  for (i in 1:10000){
    x <- mvrnorm(n = 1, mu = theta, covMat)
    max[i] = max(abs(x))
  }
  quantile[j]<- sort(max)[10000 - 10000*alpha]
  j <- j+1
} 


plot(c(0,0.5,0.9,1),quantile,type = 'l',col = 'blue') # dla jedynki wyjdzie kwantyl 1.96


####b)

#covMat <- matrix(c(1,rep(0,10)),ncol = 10,nrow = 10)
power_c <- c()
i = 1
n <-10
alpha <- 0.05
s <- 5 # to sobie sam ustalam 
kupa <- seq(0,6,by = 0.1)
cov<- c(0,0.5,0.9,1)

matrix_data <- matrix(0,ncol = 12,nrow = length(seq(0,6,by = 0.1))) # sxq

for (s in c(1,5,8)){ 
for (q in cov){
  covMat <- matrix(c(1,rep(p,n)),ncol = n,nrow = n)
  for(k in seq(0,6,by = 0.1)){
  theta <- c(rep(k,times = s),rep(0,times = n-s))
  for(j in 1:100){
    x<- mvrnorm(n=1,mu = theta,covMat)
    true_reject[j] <-sum( x[1:s] > qnorm(0.5*(1+(1-alpha)^(1/100)))) / s
  }
  power_c[i] <- sum(true_reject)/length(true_reject)
  i = i+1
  }
  
  
}
}

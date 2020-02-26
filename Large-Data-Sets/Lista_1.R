






### 2)
theta <- 0
thetas <- c(-5,-2,0,2,5)
n <- 1000000
par(mfrow = c(1,2))
for(mu in thetas){

n <- 1000000
x <- rnorm(n,mu,1)
p1 <- 2- 2*pnorm(abs(x))
p2 <- 1- pnorm(x)
plot(density(p1))
plot(density(p2))
}

##EXERCISE @ 
#zajebiście działa
t <- 3
wiersz <- c(1,rep(2.1,t+1))
k <- matrix(wiersz,t+1,t+1)
dim(k)
eigen(k)

macierz <- matrix(c(rep(1,times = 9)),nrow = 3,ncol = 3)

eigen(macierz)


#exercise 3
seq(0,1,by = 0.01)
ro <- 0.5
ile <- 10 

mtx<- list()
for(j in 1:1000){

  X <- rnorm(ile+1)
  w <- c()

for(i in 1:ile ){
  w[i]<- sqrt(ro)*X[1]+ sqrt(1-ro)*X[i+1]
}
mtx[[j]] <- w%*%t(w)
}

wynik <- matrix(0,ncol = 10,nrow = 10 )
for(i in 1:1000){
  wynik <- wynik + mtx[[i]]
}
wynik<- wynik/1000

cov(w,w)





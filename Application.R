source("INARMA_RJMCMC.R")


#Logging Industry Injury data (More et. al, 2009)
#Number of monthly claims of short-term disability benefits made by injured workers to the 
#British Columbia Worker's Compensation Board (Jan 1985 - Dec 1994)
x_data=as.numeric(t(read.table("tree.txt")))
n=length(x_data)
plot(x_data,xlab="Month",ylab="Number of claims", type="l")
#Initialise order for INARMA(p,q)
q_init=2
p_init=2

#Initialise alpha_1,...,alpha_p,beta_1,...,beta_q,lambda
alpha=rep(1/(p+1),p)
beta=rep(1/(q+1),q)
lambda=1

#Initalise augmented data
y=matrix(rep(0,n*p),byrow=TRUE,nrow=p)
v=matrix(rep(0,n*q),byrow=TRUE,nrow=q)
z=x_data

#Set maximum order
p.max=4
q.max=4

init_augs=rbind(y,v,z)
init_pars=c(alpha,beta,lambda)
init_order=c(p,q)
order_max = c(p.max, q.max)
N_reps = 5000

sample=inarma_rjmcmc(x_data,init_augs,init_pars,init_order,order_max,N_reps)

#Illustration

par(mfrow=c(2,1))

p_samp=sample$Order_samples[1,]
plot(p_samp,type="l",ylab="p",xlab="Iterations", yaxt="n")
ytick<-seq(min(p_samp),max(p_samp),by = 1)
axis(side=2, at=ytick, labels = TRUE)
q_samp=sample$Order_samples[2,]

plot(q_samp,type="l",ylab="q",xlab="Iterations", yaxt="n")
ytick<-seq(min(q_samp),max(q_samp),by = 1)
axis(side=2, at=ytick, labels = TRUE)

#Throw out B = N_reps/10 as burn-in

B=N_reps/10
par(mfrow=c(max(p_samp),1))
for(i in 1:max(p_samp)){
  alpha=sample$Parameter_samples[,i][-(1:B)]
  plot((1:N_reps)[-(1:B)],alpha,type="l",ylab=paste("alpha_",i),xlab="Iterations")
  
}
par(mfrow=c(max(q_samp),1))
for(i in 1:max(q_samp)){
  beta=sample$Parameter_samples[,(max(p_samp)+i)][-(1:B)]
  plot((1:N_reps)[-(1:B)],beta,type="l",ylab=paste("beta_",i),xlab="Iterations")
  
}
par(mfrow=c(1,1))
for(i in 1:max(q_samp)){
  lambda=sample$Parameter_samples[,(max(p_samp)+max(q_samp)+1)][-(1:B)]
  plot((1:N_reps)[-(1:B)],lambda,type="l",ylab="lambda",xlab="Iterations")
  
}


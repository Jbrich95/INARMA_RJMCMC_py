sim_aug_AR=function(x,y,z,alpha,lambda,p,rmax){
  n=length(x)
  out=matrix(0,nrow=n,ncol=(p+1))
  yprop=matrix(rep(0,p*n),byrow=TRUE,nrow=p)
  zprop=x
  
  
  
  for(t in (rmax):n){
    
    repeat{
      
      for(i in 1:p){
        yprop[i,t]=rbinom(1,x[t-i],alpha[i])
        
      }
      zprop[t]=x[t]-sum(yprop[1:p,t])
      if(zprop[t]>=0){ break
      }}
    
    uaug=runif(1)
    A=dpois(zprop[t],lambda)/dpois(z[t],lambda)
    if(uaug<=A){
      y[,t]=yprop[,t]
      
      z[t]=zprop[t]
    }
  }
    out[,1:p]=t(y)
    
    out[,(p+1)]=z
  
    return(out)
}


sim_pars_AR=function(x,y,z,p){
  n=length(x)
  alpha=rep(0,p)
  for(i in 1:p){
    alpha[i]=rbeta(1,sum(y[i,(p+1):n])+1,sum(x[(p+1-i):(n-i)]-y[i,(p+1):n])+1)
  }
  
  lambda=rgamma(1,sum(z[(p+1):n])+1,n+1)
  return (c(alpha,lambda))
}

sim_aug_MA=function(x,v,z,beta,lambda,q,rmax){
 
  n=length(x)
  aug=matrix(0,nrow=n,ncol=(q+1))
  vprop=matrix(rep(0,q*n),byrow=TRUE,nrow=q)
  zprop=x
  
  for(t in (rmax):(n-1)){
    
    repeat{
      
      for(j in 1:q){
        vprop[j,t]=rbinom(1,z[t-j],beta[j])
        
      }
      zprop[t]=x[t]-sum(vprop[1:q,t])
      if(zprop[t]>=0){ break
      }}
    
    uaug=runif(1)
    Q=min(q,(n-t))
    A=dpois(zprop[t],lambda)/dpois(z[t],lambda)
    for(i in 1:Q){
      A=A*dbinom(v[i,t+i],zprop[t],beta[i])/dbinom(v[i,t+i],z[t],beta[i])
    }
    
    
    
    if(uaug<=A){
      v[,t]=vprop[,t]
      
      z[t]=zprop[t]
    }
    
    
  }
  
  t=n
  repeat{
    
    for(j in 1:q){
      vprop[j,t]=rbinom(1,z[t-j],beta[j])
      
    }
    zprop[t]=x[t]-sum(vprop[1:q,t])
    if(zprop[t]>=0){ break
    }}
  
  u=runif(1)
  
  A=dpois(zprop[t],lambda)/dpois(z[t],lambda)
  
  if(u<=A){
    v[,t]=vprop[,t]
    
    z[t]=zprop[t]
  }
  aug[,1:q]=t(v)
  
  aug[,(q+1)]=z
  
  return (aug)
}
sim_pars_MA=function(x,v,z,q){
  n=length(x)
  
  beta=rep(0,q)
  repeat{
    for(j in 1:q){
      beta[j]=rbeta(1,sum(v[j,(q+1):n])+1,sum(z[(q+1-j):(n-j)]-v[j,(q+1):n])+1)
    }
    if(sum(beta)<1){break
    }}
  
  
  lambda=rgamma(1,sum(z[(q+1):n])+1,(n-q)+1)
  return (c(beta,lambda))
}

sim_aug_ARMA=function(x,y,v,z,alpha,beta,lambda,p,q,rmax){
  n=length(x)
  aug=matrix(0,nrow=n,ncol=(p+q+1))
  yprop=matrix(rep(0,p*n),byrow=TRUE,nrow=p)
  vprop=matrix(rep(0,q*n),byrow=TRUE,nrow=q)
  zprop=x
  
  
  
  for(t in rmax:(n-1)){
    
    repeat{
      
      for(j in 1:q){
        vprop[j,t]=rbinom(1,z[t-j],beta[j])
        
      }
      for(i in 1:p){
        yprop[i,t]=rbinom(1,x[t-i],alpha[i])
        
      }
      zprop[t]=x[t]-sum(vprop[1:q,t])-sum(yprop[1:p,t])
      if(zprop[t]>=0){ break
      }}
  
    uAug=runif(1)
    Q=min(q,(n-t))
    A=dpois(zprop[t],lambda)/dpois(z[t],lambda)
    for(i in 1:Q){
      A=A*dbinom(v[i,t+i],zprop[t],beta[i])/dbinom(v[i,t+i],z[t],beta[i])
    }
    
    
    
    if(uAug<=A){
      v[,t]=vprop[,t]
      y[,t]=yprop[,t]
      
      z[t]=zprop[t]
    }
    
    
  }
  
  t=n
  repeat{
    
    for(j in 1:q){
      vprop[j,t]=rbinom(1,z[t-j],beta[j])
      
    }
    for(i in 1:p){
      yprop[i,t]=rbinom(1,x[t-i],alpha[i])
      
    }
    zprop[t]=x[t]-sum(vprop[1:q,t])-sum(yprop[1:p,t])
    if(zprop[t]>=0){ break
    }}
  
  u=runif(1)
  
  A=dpois(zprop[t],lambda)/dpois(z[t],lambda)
  
  if(u<=A){
    v[,t]=vprop[,t]
    y[,t]=yprop[,t]
    
    z[t]=zprop[t]
  }
    
  aug[,1:p]=t(y)
  
  aug[,(p+1):(q+p)]=t(v)
  
  aug[,(p+q+1)]=z
  
  return(aug)
}

sim_pars_ARMA=function(x,y,v,z,p,q){
  n=length(x)
  
  alpha=rep(0,p)
  repeat{
    for(i in 1:p){
      alpha[i]=rbeta(1,sum(y[i,(p+1):n])+1,sum(x[(p+1-i):(n-i)]-y[i,(p+1):n])+1)
    }
    if(sum(alpha)<1){break
    }}
  
  beta=rep(0,q)
  repeat{
    for(j in 1:q){
      beta[j]=rbeta(1,sum(v[j,(q+1):n])+1,sum(z[(q+1-j):(n-j)]-v[j,(q+1):n])+1)
    }
    if(sum(beta)<1){break
    }}
  
  
  lambda=rgamma(1,sum(z[(q+1):n])+1,(n-q)+1)
  return(c(alpha,beta,lambda))
}


inarma_rjmcmc=function(x_data_data,init_augs,init_pars,init_order,order_max,N_reps){
  n=length(x_data_data)
  rmax=max(p.max,q.max)+2
  
  order_count=rbind(rep(0,N_reps),rep(0,N_reps))
  pars_sample=vector("list",N_reps)
  
  p=init_order[1]
  q=init_order[2]
  
  y = init_augs[1:p,]
  v = init_augs[(p+1):(p+q),]
  z = init_augs[-(1:(p+q)),]
  
  alpha=init_pars[1:p]
  beta=init_pars[(p+1):(p+q)]
  lambda=init_pars[-(1:(p+q))]
  for(iter in 1:N_reps){
    
    #P-order step (ar->ar)
    if(p>1){
    repeat{
      order=sample(c(-1,1),1)
      if((p!=p.max | order!=1)){
        break
      }}
    if(order==1){
      pprop=p+1
      
      U=runif(1)
      K=sample(c(1:p),1)
      
      
      alphaprop=alpha
      alphaprop[K]=U*alpha[K]
      alphaprop[pprop]=(1-U)*alpha[K]
      
      yprop=rbind(y,rep(0,n))
      S=rep(0,n)
      for(t in 1:n){
        S[t]=rbinom(1,y[K,t],U)
      }
      yprop[K,]=S
      
      yprop[pprop,]=y[K,]-S
      
      
      prob=1
      for(t in rmax:n){
        prob=prob*dbinom(yprop[K,t],x_data[t-K],alphaprop[K])*dbinom(yprop[pprop,t],x_data[t-pprop],alphaprop[pprop])/dbinom(y[K,t],x_data[t-K],alpha[K])
      }
      
      f=prob*n^(-0.5)*pprop
      
      r=1
      for(t in rmax:n){
        r=r*dbinom(yprop[K,t],y[K,t],U)
      }
      
      J=alpha[K]  
      
      A=f*(1/r)*J
      
      u1=runif(1)
      if(u1<=A){
        p=pprop
        alpha=alphaprop
        y=yprop
      }
      
    }
    if(order==-1){
      
      pprop=p-1
      
      K=sample(c(1:(p-1)),1)
      
      
      alphaprop=alpha[1:pprop]
      alphaprop[K]=alpha[K]+alpha[p]
      
      
      yprop=rbind(y[1:pprop,])
      
      yprop[K,]=y[K,]+y[p,]
      
      
      U=alpha[K]/(alphaprop[K])
      
      prob=1
      for(t in (rmax):n){
        prob=prob*dbinom(yprop[K,t],x_data[t-K],alphaprop[K])/(dbinom(y[K,t],x_data[t-K],alpha[K])*dbinom(y[p,t],x_data[t-p],alpha[p]))
      }
      
      f=prob*n^(0.5)*(1/p)
      
      r=1
      for(t in rmax:n){
        r=r*dbinom(y[K,t],yprop[K,t],U)
      }
      
      J=1/alphaprop[K]  
      
      A=f*r*J
      
      
      u2=runif(1)
      if(u2<=A){
        p=pprop
        alpha=alphaprop
        y=yprop
      }
      
    }
    }else if(p==1 ){
      #P-order step (ARMA-> MA)
      repeat{
      order=sample(c(-1,1),1)
        if(order!=-1|q!=0){
          break
          }}
      
     if(order==-1 & q!=0){
      pprop=0
      
      lambdaprop=lambda/(1-alpha[1])
      
      gamma=rep(1,q+1)
      for(k in 2:(q+1)){
        gamma[k]=beta[k-1]/(sum(beta)+1)
      }
      gamma[1]=1/(sum(beta)+1)
      
      S=matrix(rep(0,n*(q+1)),byrow=TRUE,ncol=q+1)
      
      for(t in 1:n){
        S[t,]=rmultinom(1,y[1,t],gamma)
        
      }
      zprop=rep(0,n)
      vprop=matrix(rep(0,n*q),nrow=q)
      for(t in 1:n){
        zprop[t]=z[t]+S[t,1]
        for(j in 1:q){
          vprop[j,t]=v[j,t]+S[t,j+1]
        }
      }
     
      
      prob=rep(1,n)
      for(t in rmax:n){
        prob[t]=dpois(zprop[t],lambdaprop)/(dpois(z[t],lambda)*dbinom(y[1,t],x_data_data[t-1],alpha[1]))
        for(j in 1:q){
          prob[t]=prob[t]*dbinom(vprop[j,t],zprop[t-j],beta[j])/dbinom(v[j,t],z[t-j],beta[j])
        }
      }
      f=prod(prob)*n^(0.5)
      
  U=alpha[1]
      
      r=rep(1,n)
         
      for(t in rmax:n){
      r[t]=dbinom(S[t,1],zprop[t],U)
        for(j in 1:q){
          r[t]=r[t]*dbinom(S[t,j+1],vprop[j,t],U)
        }
        r[t]=r[t]/dmultinom(S[t,],y[1,t],gamma)
      }
         
      r=prod(r)
      
      J=1/(1-U)
      
      
      A=f*r*J
    
        
        u3=runif(1)
      if(u3<=A){
        p=pprop
        
        lambda=lambdaprop
        v=vprop
        z=zprop
        y=0
        alpha=0
      }
      }
      
      if(order==1 & p!=p.max){
        pprop=p+1
      
      U=runif(1)
      K=sample(c(1:p),1)
      
      
      alphaprop=alpha
      alphaprop[K]=U*alpha[K]
      alphaprop[pprop]=(1-U)*alpha[K]
      
      yprop=rbind(y,rep(0,n))
      S=rep(0,n)
      for(t in 1:n){
        S[t]=rbinom(1,y[K,t],U)
      }
      yprop[K,]=S
      
      yprop[pprop,]=y[K,]-S
      
      
      prob=rep(1,n)
      for(t in (rmax):n){
        prob[t]=dbinom(yprop[K,t],x_data_data[t-K],alphaprop[K])*dbinom(yprop[pprop,t],x_data[t-pprop],alphaprop[pprop])/dbinom(y[K,t],x_data[t-K],alpha[K])
      }
      
      f=prod(prob)*n^(-0.5)*pprop
      
      r=1
      for(t in rmax:n){
        r=r*dbinom(yprop[K,t],y[K,t],U)
      }
      
      J=alpha[K]  
      
      A=f*(1/r)*J
      
      u4=runif(1)
      if(u4<=A){
        p=pprop
        alpha=alphaprop
        y=yprop
      }
        
      }
    }else if(p==0 & q!=0 & p!=p.max){
      #P-order step (Ma-> ARMA)
      pprop=1
      
      U=runif(1)
      
      alphaprop=U
      
      lambdaprop=lambda*(1-U)
      
      S=matrix(rep(0,n*(q+1)),ncol=q+1)
      
      for(t in 1:n){
        for(j in 2:(q+1)){
          S[t,j]=rbinom(1,v[(j-1),t],U)
        }
        S[t,1]=rbinom(1,z[t],U)
      }
      
      vprop=matrix(rep(0,n*q),nrow=q)
      yprop=rep(0,n)
      zprop=rep(0,n)
      for(t in 1:n){
        yprop[t]=sum(S[t,])
        zprop[t]=z[t]-S[t,1]
        for(j in 1:q){
          vprop[j,t]=v[j,t]-S[t,(j+1)]
        }
      }
      
      prob=rep(1,n)
      for(t in rmax:n){
        prob[t]=dpois(zprop[t],lambdaprop)*dbinom(yprop[t],x_data[t-1],alphaprop)/dpois(z[t],lambda)
        for(j in 1:q){
          prob[t]=prob[t]*dbinom(vprop[j,t],zprop[t-j],beta[j])/dbinom(v[j,t],z[t-j],beta[j])
        }
      }
      f=prod(prob)*n^(-0.5)
      
      gamma=rep(1,q+1) 
      for(j in 2:(q+1)){
        gamma[j]=beta[j-1]/(sum(beta)+1)
      }
      gamma[1]=1/(sum(beta)+1)
      
      r=rep(1,n)
      for(t in rmax:n){
        r[t]=dmultinom(S[t,],yprop[t],gamma)
        for(j in 1:q){
          r[t]=r[t]/dbinom(S[t,j+1],v[j,t],U)
        }
        r[t]=r[t]/dbinom(S[t,1],z[t],U)
      }
      r=prod(r)
      
      J=(1-U)
      
      
      A=f*r*J
      
      u5=runif(1)
      if(u5<=A){
        p=pprop
        
        alpha=alphaprop
        y=rbind(yprop)
        v=vprop
        z=zprop
      }
      
    }
    
    #Q-order step (ma->ma)
    if(q>1){
    repeat{
      order=sample(c(-1,1),1)
      if((q!=q.max | order!=1)){
        break
      }}
    if(order==1){
        qprop=q+1
        
        U=runif(1)
        K=sample(c(1:q),1)
        
        
        betaprop=beta
        betaprop[K]=U*beta[K]
        betaprop[qprop]=(1-U)*beta[K]
        
        vprop=rbind(v,rep(0,n))
        S=rep(0,n)
        for(t in 1:n){
          S[t]=rbinom(1,v[K,t],U)
        }
        vprop[K,]=S
        
        vprop[qprop,]=v[K,]-S
        
        
        prob=rep(1,n)
        for(t in (rmax):n){
          prob[t]=dbinom(vprop[K,t],z[t-K],betaprop[K])*dbinom(vprop[qprop,t],z[t-qprop],betaprop[qprop])/dbinom(v[K,t],z[t-K],beta[K])
        }
        
        f=prod(prob)*n^(-0.5)*qprop
        
        r=1
        for(t in (rmax):n){
          r=r*dbinom(vprop[K,t],v[K,t],U)
        }
        
        J=beta[K]  
        
        A=f*(1/r)*J
        
        u6=runif(1)
        if(u6<=A){
          q=qprop
          beta=betaprop
          v=vprop
        }
        
      }
    if(order==-1){
      
      qprop=q-1
      
      K=sample(c(1:(q-1)),1)
      
      
      betaprop=beta[1:qprop]
      betaprop[K]=beta[K]+beta[q]
      
      
      vprop=rbind(v[1:qprop,])
      
      vprop[K,]=v[K,]+v[q,]
      
      
      U=beta[K]/(betaprop[K])
      
      prob=1
      for(t in (rmax):n){
        prob=prob*dbinom(vprop[K,t],z[t-K],betaprop[K])/(dbinom(v[K,t],z[t-K],beta[K])*dbinom(v[q,t],z[t-q],beta[q]))
      }
      
      f=(1/q)*prob*n^(0.5)
      
      r=1
      for(t in rmax:n){
        r=r*dbinom(v[K,t],vprop[K,t],U)
      }
      
      J=1/betaprop[K]  
      
      A=f*r*J
      
      
      u7=runif(1)
      if(u7<=A){
        q=qprop
        beta=betaprop
        v=vprop
      }
      
    }
    }else if(q==1){
      #Q-order step (ARMA-> AR)
      repeat{
      order=sample(c(-1,1),1)
      if(order!=-1|p!=0){break
        }
    }
      if(order==1 & q!=q.max){
        qprop=q+1
        
        U=runif(1)
        K=sample(c(1:q),1)
        
        
        betaprop=beta
        betaprop[K]=U*beta[K]
        betaprop[qprop]=(1-U)*beta[K]
        
        vprop=rbind(v,rep(0,n))
        S=rep(0,n)
        for(t in 1:n){
          S[t]=rbinom(1,v[K,t],U)
        }
        vprop[K,]=S
        
        vprop[qprop,]=v[K,]-S
        
        
        prob=rep(1,n)
        for(t in (rmax):n){
          prob[t]=dbinom(vprop[K,t],z[t-K],betaprop[K])*dbinom(vprop[qprop,t],z[t-qprop],betaprop[qprop])/dbinom(v[K,t],z[t-K],beta[K])
        }
        
        f=prod(prob)*n^(-0.5)
        
        r=1
        for(t in rmax:n){
          r=r*dbinom(vprop[K,t],v[K,t],U)
        }
        
        J=beta[K]  
        
        A=qprop*f*(1/r)*J
        
        u8=runif(1)
        if(u8<=A){
          q=qprop
          beta=betaprop
          v=vprop
        }
      
      }
      if(order==-1){
        qprop=0
      
        lambdaprop=lambda*(1+beta[1])
        
        U=beta[1]
        
        zprop=rep(0,n)
      
      for(t in 1:n){
        zprop[t]=z[t]+v[1,t]
      }
      
        prob=rep(1,n)
        for(t in (rmax):n){
          prob[t]=dpois(zprop[t],lambdaprop)/(dpois(z[t],lambda)*dbinom(v[1,t],z[t-1],beta[1]))
        }
        
        f=prod(prob)*n^(0.5)
        
        r=1
        for(t in rmax:n){
          r=r*dbinom(v[1,t],min(zprop[t],z[t-1]),U/(1+U))
        }
        
        J=(1+U)
        A=f*r*J
      
      u9=runif(1)
      if(u9<=A){
        q=qprop
        
        lambda=lambdaprop
        
        z=zprop
        
        beta=0
        
        v=0
      }}}else if(q==0 & p!=0 & q!=q.max){
      #Q-order step (Ar-> ARMA)
      qprop=1
      U=runif(1)
      
      
      betaprop=U
      lambdaprop=lambda/(1+U)
      
      S=rep(0,n)
      S[1]=rbinom(1,z[1],U/(1+U))
        vprop=rep(0,n)
      vprop[1]=S[1]
        zprop=rep(0,n)
      zprop[1]=z[1]-S[1]
      for(t in 2:n){
        S[t]=rbinom(1,min(z[t],zprop[t-1]),U/(1+U))
        vprop[t]=S[t]
        zprop[t]=z[t]-vprop[t]
      }
      
      prob=rep(1,n)
      for(t in rmax:n){
        prob[t]=(dpois(zprop[t],lambdaprop)/dpois(z[t],lambda))*dbinom(vprop[t],zprop[t-1],betaprop[1])
      }
      
      f=prod(prob)*n^(-0.5)
      
      r=1
      for(t in rmax:n){
        r=r*dbinom(vprop[t],min(z[t],zprop[t-1]),U/(1+U))
      }
      
      J=1/(1+U)
      
      A=f*(1/r)*J
      u10=runif(1)
      
      if(u10<=A){
        q=qprop
        beta=betaprop
        lambda=lambdaprop
        v=rbind(vprop)
        z=zprop
      }
      
    }
      
    order_count[,iter]=c(p,q)

    
    if(q==0){
    theta=sim_pars_AR(x_data,y,z,p)
    alpha=theta[1:p]
    lambda=theta[p+1]
    out=sim_aug_AR(x_data,y,z,alpha,lambda,p,rmax)
    y=t(out[,1:p])
    v=0
    z=t(out[,p+1])
    
    pars_sample[[iter]]=list("Alpha"=alpha,"Beta"=NULL,"lambda"=lambda)
    }
    if(p==0){
      theta=sim_pars_MA(x_data,v,z,q)
      beta=theta[1:q]
      lambda=theta[q+1]
  
      out=sim_aug_MA(x_data,v,z,beta,lambda,q,rmax)
      v=t(out[,1:q])
      y=0
      z=t(out[,q+1])
      pars_sample[[iter]]=list("Alpha"=NULL,"Beta"=beta,"lambda"=lambda)
      
    }
    if(p!=0 & q!=0){
      theta=sim_pars_ARMA(x_data,y,v,z,p,q)
      alpha=theta[1:p]
      beta=theta[(p+1):(q+p)]
      lambda=theta[p+q+1]

      out=sim_aug_ARMA(x_data,y,v,z,alpha,beta,lambda,p,q,rmax)
      
      y=t(out[,1:p])
      v=t(out[,(p+1):(q+p)])
      z=t(out[,p+q+1])
      
      pars_sample[[iter]]=list("Alpha"=alpha,"Beta"=beta,"lambda"=lambda)
      
    }
    
  if(iter%%100 ==0){print(paste("Replication: ",iter))}
  }
  
  samp.max.p=max(order_count[1,])
  samp.max.q=max(order_count[2,])
  
  par_mat=matrix(0,nrow=N_reps,ncol=samp.max.p+samp.max.q+1)
  
  for(i in 1:N_reps){
    alphas=pars_sample[[i]]$Alpha
    betas=pars_sample[[i]]$Beta
    lambda=pars_sample[[i]]$lambda
    if(!is.null(alphas)){
    par_mat[i,1:order_count[1,i]]=alphas
    }
    if(!is.null(betas)){
      
    par_mat[i,(samp.max.p+1):(samp.max.p+order_count[2,i])]=betas
    }
    par_mat[i,samp.max.p+samp.max.q+1]=lambda
    
  }
  
  return(list("Parameter_samples"=par_mat,"Order_samples"= order_count))
}





da,p,rmax){
  n=length(x)
  out=matrix(0,nrow=n,ncol=(p+1))
  yprop=matrix(rep(0,p*n),byrow=TRUE,nrow=p)
  zprop=x
  
  
  
  for(t in (rmax):n){
    
    repeat{
      
      for(i in 1:p){
        yprop[i,t]=rbinom(1,x[t-i],alpha[i])
        
      }
      zprop[t]=x[t]-sum(yprop[1:p,t])
      if(zprop[t]>=0){ break
      }}
    
    uaug=runif(1)
    A=dpois(zprop[t],lambda)/dpois(z[t],lambda)
    if(uaug<=A){
      y[,t]=yprop[,t]
      
      z[t]=zprop[t]
    }
  }
    out[,1:p]=t(y)
    
    out[,(p+1)]=z
  
    return(out)
}


sim_pars_AR=function(x,y,z,p){
  n=length(x)
  alpha=rep(0,p)
  for(i in 1:p){
    alpha[i]=rbeta(1,sum(y[i,(p+1):n])+1,sum(x[(p+1-i):(n-i)]-y[i,(p+1):n])+1)
  }
  
  lambda=rgamma(1,sum(z[(p+1):n])+1,n+1)
  return (c(alpha,lambda))
}

sim_aug_MA=function(x,v,z,beta,lambda,q,rmax){
 
  n=length(x)
  aug=matrix(0,nrow=n,ncol=(q+1))
  vprop=matrix(rep(0,q*n),byrow=TRUE,nrow=q)
  zprop=x
  
  for(t in (rmax):(n-1)){
    
    repeat{
      
      for(j in 1:q){
        vprop[j,t]=rbinom(1,z[t-j],beta[j])
        
      }
      zprop[t]=x[t]-sum(vprop[1:q,t])
      if(zprop[t]>=0){ break
      }}
    
    uaug=runif(1)
    Q=min(q,(n-t))
    A=dpois(zprop[t],lambda)/dpois(z[t],lambda)
    for(i in 1:Q){
      A=A*dbinom(v[i,t+i],zprop[t],beta[i])/dbinom(v[i,t+i],z[t],beta[i])
    }
    
    
    
    if(uaug<=A){
      v[,t]=vprop[,t]
      
      z[t]=zprop[t]
    }
    
    
  }
  
  t=n
  repeat{
    
    for(j in 1:q){
      vprop[j,t]=rbinom(1,z[t-j],beta[j])
      
    }
    zprop[t]=x[t]-sum(vprop[1:q,t])
    if(zprop[t]>=0){ break
    }}
  
  u=runif(1)
  
  A=dpois(zprop[t],lambda)/dpois(z[t],lambda)
  
  if(u<=A){
    v[,t]=vprop[,t]
    
    z[t]=zprop[t]
  }
  aug[,1:q]=t(v)
  
  aug[,(q+1)]=z
  
  return (aug)
}
sim_pars_MA=function(x,v,z,q){
  n=length(x)
  
  beta=rep(0,q)
  repeat{
    for(j in 1:q){
      beta[j]=rbeta(1,sum(v[j,(q+1):n])+1,sum(z[(q+1-j):(n-j)]-v[j,(q+1):n])+1)
    }
    if(sum(beta)<1){break
    }}
  
  
  lambda=rgamma(1,sum(z[(q+1):n])+1,(n-q)+1)
  return (c(beta,lambda))
}

sim_aug_ARMA=function(x,y,v,z,alpha,beta,lambda,p,q,rmax){
  n=length(x)
  aug=matrix(0,nrow=n,ncol=(p+q+1))
  yprop=matrix(rep(0,p*n),byrow=TRUE,nrow=p)
  vprop=matrix(rep(0,q*n),byrow=TRUE,nrow=q)
  zprop=x
  
  
  
  for(t in rmax:(n-1)){
    
    repeat{
      
      for(j in 1:q){
        vprop[j,t]=rbinom(1,z[t-j],beta[j])
        
      }
      for(i in 1:p){
        yprop[i,t]=rbinom(1,x[t-i],alpha[i])
        
      }
      zprop[t]=x[t]-sum(vprop[1:q,t])-sum(yprop[1:p,t])
      if(zprop[t]>=0){ break
      }}
  
    uAug=runif(1)
    Q=min(q,(n-t))
    A=dpois(zprop[t],lambda)/dpois(z[t],lambda)
    for(i in 1:Q){
      A=A*dbinom(v[i,t+i],zprop[t],beta[i])/dbinom(v[i,t+i],z[t],beta[i])
    }
    
    
    
    if(uAug<=A){
      v[,t]=vprop[,t]
      y[,t]=yprop[,t]
      
      z[t]=zprop[t]
    }
    
    
  }
  
  t=n
  repeat{
    
    for(j in 1:q){
      vprop[j,t]=rbinom(1,z[t-j],beta[j])
      
    }
    for(i in 1:p){
      yprop[i,t]=rbinom(1,x[t-i],alpha[i])
      
    }
    zprop[t]=x[t]-sum(vprop[1:q,t])-sum(yprop[1:p,t])
    if(zprop[t]>=0){ break
    }}
  
  u=runif(1)
  
  A=dpois(zprop[t],lambda)/dpois(z[t],lambda)
  
  if(u<=A){
    v[,t]=vprop[,t]
    y[,t]=yprop[,t]
    
    z[t]=zprop[t]
  }
  aug[,1:p]=t(y)
  
  aug[,(p+1):(q+p)]=t(v)
  
  aug[,(p+q+1)]=z
  
  return(aug)
}

sim_pars_ARMA=function(x,y,v,z,p,q){
  n=length(x)
  
  alpha=rep(0,p)
  repeat{
    for(i in 1:p){
      alpha[i]=rbeta(1,sum(y[i,(p+1):n])+1,sum(x[(p+1-i):(n-i)]-y[i,(p+1):n])+1)
    }
    if(sum(alpha)<1){break
    }}
  
  beta=rep(0,q)
  repeat{
    for(j in 1:q){
      beta[j]=rbeta(1,sum(v[j,(q+1):n])+1,sum(z[(q+1-j):(n-j)]-v[j,(q+1):n])+1)
    }
    if(sum(beta)<1){break
    }}
  
  
  lambda=rgamma(1,sum(z[(q+1):n])+1,(n-q)+1)
  return(c(alpha,beta,lambda))
}


inarma_rjmcmc=function(x_data_data,init_augs,init_pars,init_order,order_max,N_reps){
  n=length(x_data_data)
  rmax=max(p.max,q.max)+2
  
  order_count=rbind(rep(0,N_reps),rep(0,N_reps))
  pars_sample=vector("list",N_reps)
  
  p=init_order[1]
  q=init_order[2]
  
  y = init_augs[1:p,]
  v = init_augs[(p+1):(p+q),]
  z = init_augs[-(1:(p+q)),]
  
  alpha=init_pars[1:p]
  beta=init_pars[(p+1):(p+q)]
  lambda=init_pars[-(1:(p+q))]
  for(iter in 1:N_reps){
    
    #P-order step (ar->ar)
    if(p>1){
    repeat{
      order=sample(c(-1,1),1)
      if((p!=p.max | order!=1)){
        break
      }}
    if(order==1){
      pprop=p+1
      
      U=runif(1)
      K=sample(c(1:p),1)
      
      
      alphaprop=alpha
      alphaprop[K]=U*alpha[K]
      alphaprop[pprop]=(1-U)*alpha[K]
      
      yprop=rbind(y,rep(0,n))
      S=rep(0,n)
      for(t in 1:n){
        S[t]=rbinom(1,y[K,t],U)
      }
      yprop[K,]=S
      
      yprop[pprop,]=y[K,]-S
      
      
      prob=1
      for(t in rmax:n){
        prob=prob*dbinom(yprop[K,t],x_data[t-K],alphaprop[K])*dbinom(yprop[pprop,t],x_data[t-pprop],alphaprop[pprop])/dbinom(y[K,t],x_data[t-K],alpha[K])
      }
      
      f=prob*n^(-0.5)*pprop
      
      r=1
      for(t in rmax:n){
        r=r*dbinom(yprop[K,t],y[K,t],U)
      }
      
      J=alpha[K]  
      
      A=f*(1/r)*J
      
      u1=runif(1)
      if(u1<=A){
        p=pprop
        alpha=alphaprop
        y=yprop
      }
      
    }
    if(order==-1){
      
      pprop=p-1
      
      K=sample(c(1:(p-1)),1)
      
      
      alphaprop=alpha[1:pprop]
      alphaprop[K]=alpha[K]+alpha[p]
      
      
      yprop=rbind(y[1:pprop,])
      
      yprop[K,]=y[K,]+y[p,]
      
      
      U=alpha[K]/(alphaprop[K])
      
      prob=1
      for(t in (rmax):n){
        prob=prob*dbinom(yprop[K,t],x_data[t-K],alphaprop[K])/(dbinom(y[K,t],x_data[t-K],alpha[K])*dbinom(y[p,t],x_data[t-p],alpha[p]))
      }
      
      f=prob*n^(0.5)*(1/p)
      
      r=1
      for(t in rmax:n){
        r=r*dbinom(y[K,t],yprop[K,t],U)
      }
      
      J=1/alphaprop[K]  
      
      A=f*r*J
      
      
      u2=runif(1)
      if(u2<=A){
        p=pprop
        alpha=alphaprop
        y=yprop
      }
      
    }
    }else if(p==1 ){
      #P-order step (ARMA-> MA)
      repeat{
      order=sample(c(-1,1),1)
        if(order!=-1|q!=0){
          break
          }}
      
     if(order==-1 & q!=0){
      pprop=0
      
      lambdaprop=lambda/(1-alpha[1])
      
      gamma=rep(1,q+1)
      for(k in 2:(q+1)){
        gamma[k]=beta[k-1]/(sum(beta)+1)
      }
      gamma[1]=1/(sum(beta)+1)
      
      S=matrix(rep(0,n*(q+1)),byrow=TRUE,ncol=q+1)
      
      for(t in 1:n){
        S[t,]=rmultinom(1,y[1,t],gamma)
        
      }
      zprop=rep(0,n)
      vprop=matrix(rep(0,n*q),nrow=q)
      for(t in 1:n){
        zprop[t]=z[t]+S[t,1]
        for(j in 1:q){
          vprop[j,t]=v[j,t]+S[t,j+1]
        }
      }
     
      
      prob=rep(1,n)
      for(t in rmax:n){
        prob[t]=dpois(zprop[t],lambdaprop)/(dpois(z[t],lambda)*dbinom(y[1,t],x_data_data[t-1],alpha[1]))
        for(j in 1:q){
          prob[t]=prob[t]*dbinom(vprop[j,t],zprop[t-j],beta[j])/dbinom(v[j,t],z[t-j],beta[j])
        }
      }
      f=prod(prob)*n^(0.5)
      
  U=alpha[1]
      
      r=rep(1,n)
      for(t in rmax:n){
      r[t]=dbinom(S[t,1],zprop[t],U)
        for(j in 1:q){
          r[t]=r[t]*dbinom(S[t,j+1],vprop[j,t],U)
        }
        r[t]=r[t]/dmultinom(S[t,],y[1,t],gamma)
      }
      r=prod(r)
      
      J=1/(1-U)
      
      
      A=f*r*J
    
        
        u3=runif(1)
      if(u3<=A){
        p=pprop
        
        lambda=lambdaprop
        v=vprop
        z=zprop
        y=0
        alpha=0
      }
      }
      
      if(order==1 & p!=p.max){
        pprop=p+1
      
      U=runif(1)
      K=sample(c(1:p),1)
      
      
      alphaprop=alpha
      alphaprop[K]=U*alpha[K]
      alphaprop[pprop]=(1-U)*alpha[K]
      
      yprop=rbind(y,rep(0,n))
      S=rep(0,n)
      for(t in 1:n){
        S[t]=rbinom(1,y[K,t],U)
      }
      yprop[K,]=S
      
      yprop[pprop,]=y[K,]-S
      
      
      prob=rep(1,n)
      for(t in (rmax):n){
        prob[t]=dbinom(yprop[K,t],x_data_data[t-K],alphaprop[K])*dbinom(yprop[pprop,t],x_data[t-pprop],alphaprop[pprop])/dbinom(y[K,t],x_data[t-K],alpha[K])
      }
      
      f=prod(prob)*n^(-0.5)*pprop
      
      r=1
      for(t in rmax:n){
        r=r*dbinom(yprop[K,t],y[K,t],U)
      }
      
      J=alpha[K]  
      
      A=f*(1/r)*J
      
      u4=runif(1)
      if(u4<=A){
        p=pprop
        alpha=alphaprop
        y=yprop
      }
        
      }
    }else if(p==0 & q!=0 & p!=p.max){
      #P-order step (Ma-> ARMA)
      pprop=1
      
      U=runif(1)
      
      alphaprop=U
      
      lambdaprop=lambda*(1-U)
      
      S=matrix(rep(0,n*(q+1)),ncol=q+1)
      
      for(t in 1:n){
        for(j in 2:(q+1)){
          S[t,j]=rbinom(1,v[(j-1),t],U)
        }
        S[t,1]=rbinom(1,z[t],U)
      }
      
      vprop=matrix(rep(0,n*q),nrow=q)
      yprop=rep(0,n)
      zprop=rep(0,n)
      for(t in 1:n){
        yprop[t]=sum(S[t,])
        zprop[t]=z[t]-S[t,1]
        for(j in 1:q){
          vprop[j,t]=v[j,t]-S[t,(j+1)]
        }
      }
      
      prob=rep(1,n)
      for(t in rmax:n){
        prob[t]=dpois(zprop[t],lambdaprop)*dbinom(yprop[t],x_data[t-1],alphaprop)/dpois(z[t],lambda)
        for(j in 1:q){
          prob[t]=prob[t]*dbinom(vprop[j,t],zprop[t-j],beta[j])/dbinom(v[j,t],z[t-j],beta[j])
        }
      }
      f=prod(prob)*n^(-0.5)
      
      gamma=rep(1,q+1) 
      for(j in 2:(q+1)){
        gamma[j]=beta[j-1]/(sum(beta)+1)
      }
      gamma[1]=1/(sum(beta)+1)
      
      r=rep(1,n)
      for(t in rmax:n){
        r[t]=dmultinom(S[t,],yprop[t],gamma)
        for(j in 1:q){
          r[t]=r[t]/dbinom(S[t,j+1],v[j,t],U)
        }
        r[t]=r[t]/dbinom(S[t,1],z[t],U)
      }
      r=prod(r)
      
      J=(1-U)
      
      
      A=f*r*J
      
      u5=runif(1)
      if(u5<=A){
        p=pprop
        
        alpha=alphaprop
        y=rbind(yprop)
        v=vprop
        z=zprop
      }
      
    }
    
    #Q-order step (ma->ma)
    if(q>1){
    repeat{
      order=sample(c(-1,1),1)
      if((q!=q.max | order!=1)){
        break
      }}
    if(order==1){
        qprop=q+1
        
        U=runif(1)
        K=sample(c(1:q),1)
        
        
        betaprop=beta
        betaprop[K]=U*beta[K]
        betaprop[qprop]=(1-U)*beta[K]
        
        vprop=rbind(v,rep(0,n))
        S=rep(0,n)
        for(t in 1:n){
          S[t]=rbinom(1,v[K,t],U)
        }
        vprop[K,]=S
        
        vprop[qprop,]=v[K,]-S
        
        
        prob=rep(1,n)
        for(t in (rmax):n){
          prob[t]=dbinom(vprop[K,t],z[t-K],betaprop[K])*dbinom(vprop[qprop,t],z[t-qprop],betaprop[qprop])/dbinom(v[K,t],z[t-K],beta[K])
        }
        
        f=prod(prob)*n^(-0.5)*qprop
        
        r=1
        for(t in (rmax):n){
          r=r*dbinom(vprop[K,t],v[K,t],U)
        }
        
        J=beta[K]  
        
        A=f*(1/r)*J
        
        u6=runif(1)
        if(u6<=A){
          q=qprop
          beta=betaprop
          v=vprop
        }
        
      }
    if(order==-1){
      
      qprop=q-1
      
      K=sample(c(1:(q-1)),1)
      
      
      betaprop=beta[1:qprop]
      betaprop[K]=beta[K]+beta[q]
      
      
      vprop=rbind(v[1:qprop,])
      
      vprop[K,]=v[K,]+v[q,]
      
      
      U=beta[K]/(betaprop[K])
      
      prob=1
      for(t in (rmax):n){
        prob=prob*dbinom(vprop[K,t],z[t-K],betaprop[K])/(dbinom(v[K,t],z[t-K],beta[K])*dbinom(v[q,t],z[t-q],beta[q]))
      }
      
      f=(1/q)*prob*n^(0.5)
      
      r=1
      for(t in rmax:n){
        r=r*dbinom(v[K,t],vprop[K,t],U)
      }
      
      J=1/betaprop[K]  
      
      A=f*r*J
      
      
      u7=runif(1)
      if(u7<=A){
        q=qprop
        beta=betaprop
        v=vprop
      }
      
    }
    }else if(q==1){
      #Q-order step (ARMA-> AR)
      repeat{
      order=sample(c(-1,1),1)
      if(order!=-1|p!=0){break
        }
    }
      if(order==1 & q!=q.max){
        qprop=q+1
        
        U=runif(1)
        K=sample(c(1:q),1)
        
        
        betaprop=beta
        betaprop[K]=U*beta[K]
        betaprop[qprop]=(1-U)*beta[K]
        
        vprop=rbind(v,rep(0,n))
        S=rep(0,n)
        for(t in 1:n){
          S[t]=rbinom(1,v[K,t],U)
        }
        vprop[K,]=S
        
        vprop[qprop,]=v[K,]-S
        
        
        prob=rep(1,n)
        for(t in (rmax):n){
          prob[t]=dbinom(vprop[K,t],z[t-K],betaprop[K])*dbinom(vprop[qprop,t],z[t-qprop],betaprop[qprop])/dbinom(v[K,t],z[t-K],beta[K])
        }
        
        f=prod(prob)*n^(-0.5)
        
        r=1
        for(t in rmax:n){
          r=r*dbinom(vprop[K,t],v[K,t],U)
        }
        
        J=beta[K]  
        
        A=qprop*f*(1/r)*J
        
        u8=runif(1)
        if(u8<=A){
          q=qprop
          beta=betaprop
          v=vprop
        }
      
      }
      if(order==-1){
        qprop=0
      
        lambdaprop=lambda*(1+beta[1])
        
        U=beta[1]
        
        zprop=rep(0,n)
      
      for(t in 1:n){
        zprop[t]=z[t]+v[1,t]
      }
      
        prob=rep(1,n)
        for(t in (rmax):n){
          prob[t]=dpois(zprop[t],lambdaprop)/(dpois(z[t],lambda)*dbinom(v[1,t],z[t-1],beta[1]))
        }
        
        f=prod(prob)*n^(0.5)
        
        r=1
        for(t in rmax:n){
          r=r*dbinom(v[1,t],min(zprop[t],z[t-1]),U/(1+U))
        }
        
        J=(1+U)
        A=f*r*J
      
      u9=runif(1)
      if(u9<=A){
        q=qprop
        
        lambda=lambdaprop
        
        z=zprop
        
        beta=0
        
        v=0
      }}}else if(q==0 & p!=0 & q!=q.max){
      #Q-order step (Ar-> ARMA)
      qprop=1
      U=runif(1)
      
      
      betaprop=U
      lambdaprop=lambda/(1+U)
      
      S=rep(0,n)
      S[1]=rbinom(1,z[1],U/(1+U))
        vprop=rep(0,n)
      vprop[1]=S[1]
        zprop=rep(0,n)
      zprop[1]=z[1]-S[1]
      for(t in 2:n){
        S[t]=rbinom(1,min(z[t],zprop[t-1]),U/(1+U))
        vprop[t]=S[t]
        zprop[t]=z[t]-vprop[t]
      }
      
      prob=rep(1,n)
      for(t in rmax:n){
        prob[t]=(dpois(zprop[t],lambdaprop)/dpois(z[t],lambda))*dbinom(vprop[t],zprop[t-1],betaprop[1])
      }
      
      f=prod(prob)*n^(-0.5)
      
      r=1
      for(t in rmax:n){
        r=r*dbinom(vprop[t],min(z[t],zprop[t-1]),U/(1+U))
      }
      
      J=1/(1+U)
      
      A=f*(1/r)*J
      u10=runif(1)
      
      if(u10<=A){
        q=qprop
        beta=betaprop
        lambda=lambdaprop
        v=rbind(vprop)
        z=zprop
      }
      
    }
      
    order_count[,iter]=c(p,q)

    
    if(q==0){
    theta=sim_pars_AR(x_data,y,z,p)
    alpha=theta[1:p]
    lambda=theta[p+1]
    out=sim_aug_AR(x_data,y,z,alpha,lambda,p,rmax)
    y=t(out[,1:p])
    v=0
    z=t(out[,p+1])
    
    pars_sample[[iter]]=list("Alpha"=alpha,"Beta"=NULL,"lambda"=lambda)
    }
    if(p==0){
      theta=sim_pars_MA(x_data,v,z,q)
      beta=theta[1:q]
      lambda=theta[q+1]
  
      out=sim_aug_MA(x_data,v,z,beta,lambda,q,rmax)
      v=t(out[,1:q])
      y=0
      z=t(out[,q+1])
      pars_sample[[iter]]=list("Alpha"=NULL,"Beta"=beta,"lambda"=lambda)
      
    }
    if(p!=0 & q!=0){
      theta=sim_pars_ARMA(x_data,y,v,z,p,q)
      alpha=theta[1:p]
      beta=theta[(p+1):(q+p)]
      lambda=theta[p+q+1]

      out=sim_aug_ARMA(x_data,y,v,z,alpha,beta,lambda,p,q,rmax)
      
      y=t(out[,1:p])
      v=t(out[,(p+1):(q+p)])
      z=t(out[,p+q+1])
      
      pars_sample[[iter]]=list("Alpha"=alpha,"Beta"=beta,"lambda"=lambda)
      
    }
    
  if(iter%%100 ==0){print(paste("Replication: ",iter))}
  }
  
  samp.max.p=max(order_count[1,])
  samp.max.q=max(order_count[2,])
  
  par_mat=matrix(0,nrow=N_reps,ncol=samp.max.p+samp.max.q+1)
  
  for(i in 1:N_reps){
    alphas=pars_sample[[i]]$Alpha
    betas=pars_sample[[i]]$Beta
    lambda=pars_sample[[i]]$lambda
    if(!is.null(alphas)){
    par_mat[i,1:order_count[1,i]]=alphas
    }
    if(!is.null(betas)){
      
    par_mat[i,(samp.max.p+1):(samp.max.p+order_count[2,i])]=betas
    }
    par_mat[i,samp.max.p+samp.max.q+1]=lambda
    
  }
  
  return(list("Parameter_samples"=par_mat,"Order_samples"= order_count))
}






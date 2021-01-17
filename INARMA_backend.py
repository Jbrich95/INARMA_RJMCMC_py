import numpy as np
from scipy.stats import poisson
from scipy.stats import binom
from scipy.stats import beta
from scipy.stats import gamma
from scipy.stats import multinomial as multinom


#For simulating augmented variables
def sim_aug_AR(x,y,z,alphas,lam,p,rmax):
    n=len(x)

    yprop=np.zeros([n,p],dtype = int)
    zprop = np.zeros([n,1],dtype = int)
    zprop[0:(rmax-1)] = x[0:(rmax-1)]

    for t in range(rmax-1,n):

        while True:

            for i in range(0,p):
                yprop[t,i] = binom(n=x[t-i-1],p=alphas[i]).rvs()

            zprop[t]=x[t]-np.sum(yprop[t,])
            if zprop[t]>=0: 
                break

        uaug = np.random.rand()
        dist=poisson(mu=lam)
        A=dist.pmf(zprop[t])/dist.pmf(z[t])
        if uaug <= A :
            y[t,] = yprop
            z[t,] = zprop

    
def sim_aug_MA(x,v,z,betas,lam,q,rmax):

    n=len(x)

    vprop=np.zeros([n,q],dtype = int)
    zprop = np.zeros([n,1],dtype = int)
    zprop[0:(rmax-1)] = x[0:(rmax-1)]

    for t in range(rmax-1,n):

        while True:

            for i in range(0,q):
                vprop[t,i] = binom(n=z[t-i-1],p=betas[i]).rvs()

            zprop[t]=x[t]-np.sum(vprop[t,])
            if zprop[t]>=0: 
                break

        uaug = np.random.rand()
        Q = np.min([q, n - t ])
        dist=poisson(mu=lam)
        A=dist.pmf(zprop[t])/dist.pmf(z[t])
        if Q == 0:
            for i in range(0,Q):
                distprop=binom(n=zprop[t], p=betas[i])
                dist=binom(n=z[t], p=betas[i])
                np.seterr(invalid='ignore')

                A *= distprop.pmf(v[t+i+1,i])/dist.pmf(v[t+i+1,i])
            
        if uaug <= A :
            v[t,] = vprop
            z[t,] = zprop


def sim_aug_ARMA(x,y,v,z,alphas,betas,lam,p,q,rmax):
    n=len(x)

    for t in range(rmax-1,n-1):

        sim_aug_ARMA_one_time(x,y,v,z,t,alphas,betas,lam,p,q,rmax)
            
def sim_aug_ARMA_one_time(x,y,v,z,t,alphas,betas,lam,p,q,rmax):
    n=len(x)

    yprop=np.zeros(p,dtype = int)
    vprop=np.zeros(q,dtype = int)

    while True:

        for i in range(0,p):
            yprop[i] = binom(n=x[t-i-1],p=alphas[i]).rvs()
        for j in range(0,q):
            vprop[j] = binom(n=z[t-j-1],p=betas[j]).rvs()
        zprop=x[t]-np.sum(yprop)-np.sum(vprop)
        if zprop>=0: 
            break

    uaug = np.random.rand()
    Q = np.min([q, n - t])
    dist=poisson(mu=lam)
    A=dist.pmf(zprop)/dist.pmf(z[t])
    if Q == 0:
        for i in range(0,Q):
            distprop=binom(n=zprop, p=betas[i])
            dist=binom(n=z[t], p=betas[i])
            np.seterr(invalid='ignore')

            A *= distprop.pmf(v[t+i+1,i])/dist.pmf(v[t+i+1,i])

    if uaug <= A :
        v[t,] = vprop
        y[t,] = yprop

        z[t,] =zprop
    
#For simulating parameters
def sim_pars_AR(x,y,z,p):
    n=len(x)
    alpha = np.zeros([p,1])
    while True:
        for i in range(0,p):
            dist = beta(a=np.sum(y[p:n,i])+1,b=np.sum(x[(p-i):(n-i)])-np.sum(y[(p):n,i])+1)
            alpha[i]= dist.rvs(1)
        if sum(alpha) < 1:
            break
    lam=gamma(a=np.sum(z[p:n])+1,scale=1/(n+1)).rvs()
    return alpha,lam

def sim_pars_MA(x,v,z,q):
    n=len(x)
    beta_sample = np.zeros([q,1])
    while True:

        for i in range(0,q):
            dist = beta(a=np.sum(v[q:n,i])+1,b=np.sum(z[(q-i):(n-i)])-np.sum(v[q:n,i])+1)
            beta_sample[i]= dist.rvs(1)
        if sum(beta_sample) < 1: 
            break
  
    lam=gamma(a=sum(z[q:n])+1,scale=1/(n-q+1)).rvs()
    return beta_sample,lam


def sim_pars_ARMA(x,y,v,z,p,q):
    n=len(x)
    alpha=np.zeros(p)

    beta_sample = np.zeros(q)
    
    while True:

        for i in range(0,p):
            dist = beta(a=np.sum(y[p:n,i])+1,b=np.sum(x[(p-i):(n-i)])-np.sum(y[p:n,i])+1)
            alpha[i]= dist.rvs(1)
        if sum(alpha) < 1: 
            break
            
    while True:

        for i in range(0,q):
            dist = beta(a=np.sum(v[q:n,i])+1,b=np.sum(z[(q-i):(n-i)])-np.sum(v[q:n,i])+1)
            beta_sample[i]= dist.rvs(1)
        if sum(beta_sample) < 1: 
            break
  
    lam=gamma(a=np.sum(z[q:n])+1,scale=1/(n-q+1)).rvs()
    return alpha,beta_sample,lam



#ORDER CHANGING - p


def p_up(x_data,y,alphas,p,rmax):  
    n = len(x_data)
    pprop=p+1
    p_new=np.copy(p)

    U=np.random.rand(1)
    K=np.random.randint(0,p)
    alphaprop=np.zeros(pprop)
    alphaprop[0:p]=alphas
    alphaprop[K]=U*alphas[K]

    alphaprop[pprop-1]=(1-U)*alphas[K]
    yprop=np.zeros([n,pprop],dtype = int)
    yprop[:,0:p]=y
    S=np.array(list(map(lambda n : binom(n=n,p=U).rvs(), y[:,K])),dtype = int)

    yprop[:,pprop-1]=y[:,K]-S

    logprob=0

    for t in range(rmax-1,n):
        logprob += binom(n=x_data[t-K-1],p=alphaprop[K]).logpmf(yprop[t,K])+binom(n=x_data[t-pprop-1],p=alphaprop[pprop-1]).logpmf(yprop[t,pprop-1])-binom(n=x_data[t-K-1],p=alphas[K]).logpmf(yprop[t,K])-binom(n=y[t,K],p=U).logpmf(yprop[t,K])
    logf=logprob+np.log(pprop)-0.5*np.log(n)


    J=alphas[K]  

    logA=logf+np.log(J)

    if np.log(np.random.rand()) <= logA:
        p_new=pprop
        alphas=alphaprop
        y=yprop
    return p_new, alphas, y

def p_down(x_data,y,alphas,p,rmax):  
    n = len(x_data)
    p_new=np.copy(p)

    pprop=p-1

    U=np.random.rand(1)
    K=np.random.randint(0,p-1)
    alphaprop=alphas[0:pprop]
    yprop=y[:,0:pprop]

    yprop[:,K]=y[:,K]+y[:,p-1]
    U = alphas[K]/(alphaprop[K])

    logprob=0
    for t in range(rmax-1,n):
        logprob += binom(n=x_data[t-K-1],p=alphaprop[K]).logpmf(yprop[t,K-1])-binom(n=x_data[t-K-1],p=alphas[K]).logpmf(y[t,K])-binom(n=x_data[t-p],p=alphas[p-1]).logpmf(y[t,p-1])-binom(n=yprop[t,K],p=U).logpmf(y[t,K])

    J = 1/alphaprop[K]  
    logf=logprob+0.5*np.log(n)-np.log(p)
    logA=logf+np.log(J)

    
    if np.log(np.random.rand(1)) <= logA:
        p_new=pprop
        alphas=alphaprop
        y=yprop
    return p_new, alphas, y

def arma_to_ma(x_data,y,v,z,alphas,betas,lam,p,q,rmax):  
    n = len(x_data)
    p_new=np.copy(p)

    lamprop = lam/(1-alphas[0])

    gamma = np.ones(q+1)
    for i in range(1,q+1):
        gamma[i] = betas[i-1]/(sum(betas)+1)
    gamma[0]=1/(np.sum(betas)+1)

    S = np.zeros([n,q+1],dtype = int)
    zprop = np.zeros([n,1],dtype = int)
    vprop = np.zeros([n,q],dtype = int)
    for t in range(0,n):
        S[t,]=multinom( n = y[t,0],p= gamma).rvs(1)
        zprop[t] = z[t] + S[t,0]
        for j in range(0,q):
            vprop[t,j] = v[t,j] + S[t,j+1]


    U = alphas[0]

    logprob=0
    for t in range(rmax-1,n):
        logprob += poisson(mu=lamprop).logpmf(zprop[t])-poisson(mu=lam).logpmf(z[t])+binom(n=x_data[t-1],p=alphas[0]).logpmf(y[t,0])
        logprob += binom(n=zprop[t],p=U).logpmf(S[t,0])-multinom(n=y[t,0],p=gamma).logpmf(S[t,])
        for j in range(0,q):
            logprob += binom(n=zprop[t-j-1],p=betas[j]).logpmf(vprop[t,j])-binom(n=z[t - j - 1],p=betas[j]).logpmf(v[t,j])
            logprob +=binom(n=vprop[t,j],p=U).logpmf(S[t,j+1])

    logf=logprob+0.5*np.log(n)

    J=1/(1-U)  

    logA=logf+np.log(J)
    
    if np.log(np.random.rand()) <= logA:
        p_new=0
        lam = lamprop
        y=0
        alphas = 0
    return p_new, alphas,lam, y

def ma_to_arma(x_data, y, v,z,alphas, betas, lam,p, q,rmax)   :
    n = len(x_data)
    p_new=np.copy(p)
    U=np.random.rand(1)
    alphaprop=U
    lamprop = lam*(1-U)

    S=np.zeros([n,q+1],dtype = int)

    for t in range(0,n):
        for j in range(1,q+1):

            S[t,j]=binom(n=v[t,j-1],p=U).rvs(1)
        S[t,1]=binom(n=z[t],p=U).rvs(1)


    vprop = np.zeros([n,q],dtype = int)
    yprop = np.zeros([n,1],dtype = int)
    zprop = np.zeros([n,1],dtype = int)

    for t in range(0,n):
        yprop[t]=np.sum(S[t,])
        zprop[t]=z[t] - S[t,0]
        for j in range(0,q):
            vprop[t,j]=v[t,j] - S[t,j-1]

    logprob=1
    gamma = np.ones(q+1)

    for i in range(1,q+1) :
        gamma[i] = betas[i-1]/(np.sum(betas)+1)
    gamma[0]=1/(np.sum(betas)+1)

    for t in range(rmax-1,n):
        logprob += poisson(mu=lamprop).logpmf(zprop[t])+binom(n=x_data[t-1],p=alphaprop).logpmf(yprop[t])-poisson(mu=lam).logpmf(z[t])
        logprob +=multinom(n=yprop[t],p=gamma).logpmf(S[t,])
        for j in range(0,q):
            logprob += binom(n=zprop[t-j-1],p=betas[j]).logpmf(vprop[t,j])-binom(n=zprop[t-j-1],p=betas[j]).logpmf(v[t,j])
            logprob += -binom(n=z[t],p=U).logpmf(S[t,0])

    logf=logprob-0.5*np.log(n)

    J=(1-U)

    logA=logf+np.log(J)
    if np.log(np.random.rand(1)) <= logA:
        p_new=1
        alphas=alphaprop
        y=yprop
        v=vprop
        z=zprop
    
    return (p_new,alphas,y,v,z)

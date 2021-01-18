import numpy as np
from scipy.stats import poisson
from scipy.stats import binom
from scipy.stats import beta
from scipy.stats import gamma
from scipy.stats import multinomial as multinom
from numba import jit

#Random variable pmfs
@jit
def factorial(k):
    pr = 1
    for l in range(1, k+1):
        pr *= l
    return pr
@jit
def choose(n,k):
    return factorial(n)/(factorial(n-k)*factorial(k))
@jit
def binom_logpmf(k,n,p):
    return np.log(choose(n,k)*p**k*(1-p)**(n-k))
@jit
def poisson_logpmf(k, lam):
    return k*np.log(lam)-lam - np.log(factorial(k))
#For simulating augmented variables


def sim_aug_AR(x,v,z,alphas,lam,p,q,rmax):
    n=len(x)

    [sim_aug_AR_one_time(x,y,z,t,betas,lam,p,rmax) for t in range(rmax-1,n)]
    

def sim_aug_AR_one_time(x,y,z,t,alphas,lam,p,rmax):
    n=len(x)

    yprop=np.zeros(p,dtype = int)



    while True:

        for i in range(0,p):
            yprop[i] =  binom.rvs(n=x[t-i-1,0],p=alphas[i])

        zprop[t]=x[t]-np.sum(yprop)
        if zprop[t]>=0: 
            break

    uaug = np.random.rand()
    A=poisson_logpmf(k=zprop[0],lam=lam)-poisson_logpmf(k=z[t,0],lam=lam)
    if np.log(uaug) <= A :
        y[t,] = yprop
        z[t,0] = zprop
        

def sim_aug_MA(x,y,v,z,alphas,betas,lam,p,q,rmax):
    n=len(x)

    [sim_aug_MA_one_time(x,v,z,t,betas,lam,q,rmax) for t in range(rmax-1,n)]
    
    
def sim_aug_MA_one_time(x,v,z,t,betas,lam,q,rmax):

    n=len(x)

    vprop=np.zeros(q,dtype = int)

    while True:

        for i in range(0,q):
            vprop[j] = binom.rvs(n=z[t-j-1,0],p=betas[j])

        zprop=x[t]-np.sum(vprop)
        if zprop>=0: 
            break

    uaug = np.random.rand()
    Q = np.min([q, n - t ])
    A=poisson_logpmf(k=zprop[0],lam=lam)-poisson_logpmf(k=z[t,0],lam=lam)
    if Q == 0:
        for i in range(0,Q):

            np.seterr(invalid='ignore')

            A += binom_logpmf(k=v[t+i+1,i],n=zprop, p=betas[i])-binom_logpmf(k=v[t+i+1,i],n=z[t,0], p=betas[i])

            
    if np.log(uaug) <= A :
        v[t,] = vprop
        z[t,0] = zprop

def sim_aug_ARMA(x,y,v,z,alphas,betas,lam,p,q,rmax):
    n=len(x)

    [sim_aug_ARMA_one_time(x,y,v,z,t,alphas,betas,lam,p,q,rmax) for t in range(rmax-1,n)]
    
            
def sim_aug_ARMA_one_time(x,y,v,z,t,alphas,betas,lam,p,q,rmax):
    n=len(x)

    yprop=np.zeros(p,dtype = int)
    vprop=np.zeros(q,dtype = int)

    while True:

        for i in range(0,p):
            yprop[i] = binom.rvs(n=x[t-i-1,0],p=alphas[i])
        for j in range(0,q):
            vprop[j] = binom.rvs(n=z[t-j-1,0],p=betas[j])
        zprop=x[t]-np.sum(yprop)-np.sum(vprop)
        if zprop>=0: 
            break

    uaug = np.random.rand()
    Q = np.min([q, n - t])
                    
    A=poisson_logpmf(k=zprop[0],lam=lam)-poisson_logpmf(k=z[t,0],lam=lam)
    if Q == 0:
        for i in range(0,Q):
        
            np.seterr(invalid='ignore')

            A += binom_logpmf(k=v[t+i+1,i],n=zprop, p=betas[i])-binom_logpmf(k=v[t+i+1,i],n=z[t,0], p=betas[i])

    if np.log(uaug) <= A :
        v[t,] = vprop
        y[t,] = yprop

        z[t,0] =zprop
    
#For simulating parameters

def sim_pars_AR(x,y,z,p):
    n=len(x)
    while True:
        alphas=np.array([beta.rvs(a=np.sum(y[p:(n),i])+1,b=np.sum(x[(p-i-1):(n-i)])-np.sum(y[(p):(n),i])+1) for i in range(0,p)]).reshape(p)
        if np.sum(alphas) < 1:
            break
    lam=gamma.rvs(a=np.sum(z[p:(n)])+1,scale=1/(n+1))
    return alphas,lam


def sim_pars_MA(x,v,z,q):
    n=len(x)
    while True:

        betas=np.array([beta.rvs(a=np.sum(v[q:n,i])+1,b=np.sum(z[(q-i-1):(n-i)])-np.sum(v[(q):(n),i])+1) for i in range(0,q)]).reshape(q)       
        if sum(betas) < 1: 
            break
  
    lam=gamma.rvs(a=np.sum(z[q:(n)])+1,scale=1/(n-q+1))
    return betas,lam


def sim_pars_ARMA(x,y,v,z,p,q):
    n=len(x)
    
    while True:
        alphas=np.array([beta.rvs(a=np.sum(y[p:(n),i])+1,b=np.sum(x[(p-i-1):(n-i)])-np.sum(y[(p):(n),i])+1) for i in range(0,p)]).reshape(p)
        if np.sum(alphas) < 1:
            break
            
    while True:

        betas=np.array([beta.rvs(a=np.sum(v[q:(n),i])+1,b=np.sum(z[(q-i-1):(n-i)])-np.sum(v[(q):(n),i])+1) for i in range(0,q)]).reshape(q)       
        if sum(betas) < 1: 
            break
  
    lam=gamma.rvs(a=np.sum(z[q:(n-1)])+1,scale=1/(n-q+1))
    return alphas,betas,lam



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
    S=np.array([binom.rvs(n=n,p=U,size = 1) for n in y[:,K]]).reshape(n)
    yprop[:,pprop-1]=y[:,K]-S

    logprob=0


    logprob +=np.sum([binom_logpmf(k=k,n=n, p=alphaprop[K]) for n,k in zip(x_data[(rmax-2-K):(n-K-1),0],yprop[(rmax-1):n,K])])

    logprob +=np.sum([binom_logpmf(k=k,n=n, p=alphaprop[pprop-1]) for n,k in zip(x_data[(rmax-1-pprop):(n-pprop),0],yprop[(rmax-1):(n),pprop-1])])

    logprob -=np.sum([binom_logpmf(k=k,n=n, p=alphas[K]) for n,k in zip(x_data[(rmax-2-K):(n-K-1),0],y[(rmax-1):(n),K])])

    logprob -=np.sum([binom_logpmf(k=k,n=n, p=U) for n,k in zip(yprop[(rmax-1):(n),K],y[(rmax-1):(n),K])])
               
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

    K=np.random.randint(0,p-1)
    alphaprop=alphas[0:pprop]
    alphaprop[K]=alphas[K]+alphas[p-1]
    yprop=y[:,0:pprop]

    yprop[:,K]=y[:,K]+y[:,p-1]
    U = alphas[K]/(alphaprop[K])
    
    logprob=0
    
    logprob +=np.sum([binom_logpmf(k=k,n=n, p=alphaprop[K]) for n,k in zip(x_data[(rmax-2-K):(n-K-1),0],yprop[(rmax-1):(n),K-1])])

    logprob -=np.sum([binom_logpmf(k=k,n=n, p=alphas[K]) for n,k in zip(x_data[(rmax-2-K):(n-K-1),0],y[(rmax-1):(n),K])])

    logprob -=np.sum([binom_logpmf(k=k,n=n, p=alphas[p-1]) for n,k in zip(x_data[(rmax-1-p):(n-p),0],y[(rmax-1):(n),p-1])])

    logprob +=np.sum([binom_logpmf(k=k,n=n, p=U) for n,k in zip(yprop[(rmax-1):(n),K],y[(rmax-1):(n),K])])
        
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
    S=np.array([multinom.rvs(n=n,p=gamma,size = 1) for n in y[:,0]])
    S=S.reshape([n,q+1])
    zprop = np.zeros([n,1],dtype = int)
    vprop = np.zeros([n,q],dtype = int)
  
    for t in range(0,n):
        zprop[t,0] = z[t,0] + S[t,0]
        for j in range(0,q):
            vprop[t,j] = v[t,j] + S[t,j+1]
    

    U = alphas[0]

    logprob=0

    logprob +=np.sum([poisson_logpmf(k=k, lam = lamprop) for k in zprop[(rmax-1):(n),0]])-np.sum([poisson_logpmf(k=k, lam = lam) for k in z[(rmax-1):(n),0]])
    
    logprob -=np.sum([binom_logpmf(k=k,n=n, p=alphas[0]) for n,k in zip(x_data[(rmax-2):(n-1),0],y[(rmax-1):(n),0])])
    logprob +=np.sum([binom_logpmf(k=k,n=n, p=U) for n,k in zip(zprop[(rmax-1):(n),0],S[(rmax-1):(n),0])])
    logprob -=np.sum([multinom.logpmf(x=x,n=n, p=gamma) for n,x in zip(y[(rmax-1):(n),0],S[(rmax-1):(n),])])
                                                                                                 
                                                                                    
    for j in range(0,q):
        logprob +=np.sum([binom_logpmf(k=k,n=n, p=betas[j]) for n,k in zip(zprop[(rmax-2-j):(n-1-j),0],vprop[(rmax-1):(n),j])])
        logprob -=np.sum([binom_logpmf(k=k,n=n, p=betas[j]) for n,k in zip(z[(rmax-2-j):(n-1-j),0],v[(rmax-1):(n),j])])
        logprob +=np.sum([binom_logpmf(k=k,n=n, p=U) for n,k in zip(vprop[(rmax-1):(n),j],S[(rmax-1):(n),j+1])])


    logf=logprob+0.5*np.log(n)

    J=1/(1-U)  

    logA=logf+np.log(J)
    
    if np.log(np.random.rand(1)) <= logA:
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

    S[:,1]=np.array([binom.rvs(n=n,p=U,size = 1) for n in z]).reshape([n,1])

 
    for j in range(1,q+1):

        S[:,j]=np.array([binom.rvs(n=n,p=U,size = 1) for n in v[:,j-1]]).reshape([n,1])



    vprop = np.zeros([n,q],dtype = int)
    yprop = np.zeros([n,1],dtype = int)
    zprop = np.zeros([n,1],dtype = int)

    for t in range(0,n):
        yprop[t]=np.sum(S[t,])
        zprop[t]=z[t] - S[t,0]
        for j in range(0,q):
            vprop[t,j]=v[t,j] - S[t,j-1]

  
    gamma = np.ones(q+1)

    for i in range(1,q+1) :
        gamma[i] = betas[i-1]/(np.sum(betas)+1)
    gamma[0]=1/(np.sum(betas)+1)

            
    logprob=0
    logprob +=np.sum([poisson_logpmf(k=k, lam = lamprop) for k in zprop[(rmax-1):(n)]])-np.sum([poisson_logpmf(k=k, lam = lam) for k in z[(rmax-1):(n)]])
    logprob +=np.sum([multinom.logpmf(x=x,n=n, p=gamma) for n,x in zip(yprop[(rmax-1):(n),0],S[(rmax-1):(n),0])])
    logprob -=np.sum([binom_logpmf(k=k,n=n, p=alphaprop) for n,k in zip(x_data[(rmax-2):(n-1),0],yprop[(rmax-1):(n),0])])
    logprob -=np.sum([binom_logpmf(k=k,n=n, p=U) for n,k in zip(z[(rmax-1):(n),0],S[(rmax-1):(n),0])])
                                                                                                                                                                    
    for j in range(0,q):
        logprob +=np.sum([binom_logpmf(k=k,n=n, p=betas[j]) for n,k in zip(zprop[(rmax-2-j):(n-1-j),0],vprop[(rmax-1):(n),j])])
        logprob -=np.sum([binom_logpmf(k=k,n=n, p=betas[j]) for n,k in zip(z[(rmax-2-j):(n-1-j),0],v[(rmax-1):(n),j])])
        logprob -=np.sum([binom_logpmf(k=k,n=n, p=U) for n,k in zip(v[(rmax-1):(n),j-1],S[(rmax-1):(n),j])])

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

import numpy as np
from scipy.stats import poisson
from scipy.stats import binom
from scipy.stats import beta
from scipy.stats import gamma
from scipy.stats import multinomial as multinom


#For simulating augmented variables
def sim_aug_AR(x,y,z,alphas,lam,p,rmax):
    n=len(x)
    out=np.zeros([n,p+1])

    yprop=np.zeros([n,p])
    zprop = np.zeros([n,1])
    zprop[0:(rmax-1)] = x[0:(rmax-1)]

    for t in range(rmax-1,n):

        while True:

            for i in range(0,p):
                yprop[t,i] = binom(x[t-i],alphas[i]).rvs()

            zprop[t]=x[t]-sum(yprop[t,0:p-1])
            if zprop[t]>=0: 
                break

        uaug = np.random.rand()
        dist=poisson(lam)
        A=dist.pmf(zprop[t])/dist.pmf(z[t])
        if uaug <= A :
            y[t,] = np.copy(yprop[t,])
            z[t,] = np.copy(zprop[t,])

    out[:,0:p] = np.copy(y)
    out[:,[p]] = np.copy(z)
        
    return(out)
    
def sim_aug_MA(x,v,z,betas,lam,q,rmax):

    n=len(x)
    out=np.zeros([n,q+1])

    vprop=np.zeros([n,q])
    zprop = np.zeros([n,1])
    zprop[0:(rmax-1)] = x[0:(rmax-1)]

    for t in range(rmax-1,n):

        while True:

            for i in range(0,q):
                vprop[t,i] = binom(z[t-i],betas[i]).rvs()

            zprop[t]=x[t]-sum(vprop[t,0:q-1])
            if zprop[t]>=0: 
                break

        uaug = np.random.rand()
        Q = np.min([q, n - t + 1])
        dist=poisson(lam)
        A=dist.pmf(zprop[t])/dist.pmf(z[t])
        if Q == 0:
            for i in range(0,Q):
                distprop=binom(zprop[t], betas[i])
                dist=binom(z[t], betas[i])
                np.seterr(invalid='ignore')

                A *= distprop.pmf(v[t+i,i])/dist.pmf(v[t+i,i])
            
        if uaug <= A :
            v[t,] = np.copy(vprop[t,])
            z[t,] = np.copy(zprop[t,])

    out[:,0:q] = np.copy(v)
    out[:,[q]] = np.copy(z)
        
    return out

def sim_aug_ARMA(x,y,v,z,alphas,betas,lam,p,q,rmax):
    n=len(x)
    out=np.zeros([n,p+q+1])

    yprop=np.zeros([n,p])
    vprop=np.zeros([n,q])
    zprop = np.zeros([n,1])
    zprop[0:(rmax-1)] = x[0:(rmax-1)]

    for t in range(rmax-1,n):

        while True:

            for i in range(0,p):
                yprop[t,i] = binom(x[t-i],alphas[i]).rvs()
            for j in range(0,q):
                vprop[t,j] = binom(int(z[t-j]),betas[j]).rvs()
            zprop[t]=x[t]-sum(yprop[t,0:p-1])-sum(vprop[t,0:q-1])
            if zprop[t]>=0: 
                break

        uaug = np.random.rand()
        Q = np.min([q, n - t + 1])
        dist=poisson(lam)
        A=dist.pmf(zprop[t])/dist.pmf(z[t])
        if Q == 0:
            for i in range(0,Q):
                distprop=binom(zprop[t], betas[i])
                dist=binom(z[t], beta[i])
                np.seterr(invalid='ignore')

                A *= distprop.pmf(v[t+i,i])/dist.pmf(v[t+i,i])
            
        if uaug <= A :
            v[t,] = np.copy(vprop[t,])
            y[t,] = np.copy(yprop[t,])

            z[t,] = np.copy(zprop[t,])
            
    
    out[:,0:p] = np.copy(y)
    out[:,(p):(p+q)] = np.copy(v)
    out[:,(p+q):(p+q+1)] = np.copy(z)
    
    return out

#For simulating parameters
def sim_pars_AR(x,y,z,p):
    n=len(x)
    alpha = np.zeros([p,1])
    while True:
        for i in range(0,p):
            dist = beta(a=sum(y[(p+1):n,i])+1,b=sum(x[(p+1-i):(n-i)])-sum(y[(p+1):n,i])+1)
            alpha[i]= dist.rvs(1)
        if sum(alpha) < 1:
            break
    lam=gamma(sum(z[(p+1):n])+1,n+1).rvs()
    return alpha,lam

def sim_pars_MA(x,v,z,q):
    n=len(x)
    beta_sample = np.zeros([q,1])
    while True:

        for i in range(0,q):
            dist = beta(a=sum(v[(q+1):n,i])+1,b=sum(z[(q+1-i):(n-i)])-sum(v[(q+1):n,i])+1)
            beta_sample[i]= dist.rvs(1)
        if sum(beta_sample) < 1: 
            break
  
    lam=gamma(sum(z[(q+1):n])+1,n-q+1).rvs()
    return beta_sample,lam


def sim_pars_ARMA(x,y,v,z,p,q):
    n=len(x)
    alpha=np.zeros(p)

    beta_sample = np.zeros(q)
    
    while True:

        for i in range(0,p):
            dist = beta(a=sum(y[(p+1):n,i])+1,b=sum(x[(p+1-i):(n-i)])-sum(y[(p+1):n,i])+1)
            alpha[i]= dist.rvs(1)
        if sum(alpha) < 1: 
            break
            
    while True:

        for i in range(0,q):
            dist = beta(a=sum(v[(q+1):n,i])+1,b=sum(z[(q+1-i):(n-i)])-sum(v[(q+1):n,i])+1)
            beta_sample[i]= dist.rvs(1)
        if sum(beta_sample) < 1: 
            break
  
    lam=gamma(sum(z[(q+1):n])+1,n-q+1).rvs()
    return alpha,beta_sample,lam



#ORDER CHANGING - p


def p_up(x_data,y,alphas,p,rmax):  
    n = len(x_data)
    pprop=p+1

    U=np.random.rand(1)
    K=np.random.randint(0,p)
    alphaprop=np.zeros(pprop)
    alphaprop[0:p]=alphas
    alphaprop[K]=U*alphas[K]

    alphaprop[pprop-1]=(1-U)*alphas[K]
    yprop=np.zeros([n,pprop])
    yprop[:,0:p]=y
    S=np.zeros(n)
    for t in range(0,n):

        S[t]=binom(y[t,K],U).rvs(1)

    yprop[:,K]= S
    yprop[:,pprop-1]=y[:,K]-S

    prob=1

    for t in range(rmax-1,n):
        prob *= binom(x_data[t-K-1],alphaprop[K]).pmf(yprop[t,K])*binom(x_data[t-pprop-1],alphaprop[pprop-1]).pmf(yprop[t,pprop-1])/(binom(x_data[t-K],alphas[K]).pmf(yprop[t,K])*binom(y[t,K],U).pmf(yprop[t,K]))
    f=prob*pprop*n**(-0.5)


    J=alphas[K]  

    A=f*J

    if np.random.rand(1) <= A:
        p=pprop
        alphas=alphaprop
        y=yprop
    return p, alphas, y

def p_down(x_data,y,alphas,p,rmax):  
    n = len(x_data)

    pprop=p-1

    U=np.random.rand(1)
    K=np.random.randint(0,p-1)
    alphaprop=alphas[0:pprop]
    yprop=y[:,0:pprop]

    yprop[:,K]=y[:,K]+y[:,p-1]
    U = alphas[K]/(alphaprop[K])

    prob=1
    for t in range(rmax-1,n):
        prob *= binom(x_data[t-K-1],alphaprop[K]).pmf(yprop[t,K-1])/(binom(x_data[t-K-1],alphas[K]).pmf(y[t,K])*binom(x_data[t-p],alphas[p-1]).pmf(y[t,p-1]))*binom(yprop[t,K],U).pmf(y[t,K])

    J=1/alphaprop[K]  
    f=prob*n**(0.5)/p
    A=f*J

    
    if np.random.rand(1) <= A:
        p=pprop
        alphas=alphaprop
        y=yprop
    return p, alphas, y

def arma_to_ma(x_data,y,v,z,alphas,betas,lam,p,q,rmax):  
    n = len(x_data)

    lamprop = lam/(1-alphas[0])

    gamma = np.ones(q+1)
    for i in range(1,q+1):
        gamma[i] = betas[i-1]/(sum(betas)+1)
    gamma[0]=1/(sum(betas)+1)

    S = np.zeros([n,q+1])
    zprop = np.zeros([n,1])
    vprop = np.zeros([n,q])
    for t in range(0,n):
        S[t,]=multinom( n = y[t,0],p= gamma).rvs(1)
        zprop[t] = z[t] + S[t,0]
        for j in range(0,q):
            vprop[t,j] = v[t,j] + S[t,j+1]


    U = alphas[0]

    prob=1
    for t in range(rmax-1,n):
        prob *= poisson(lamprop).pmf(zprop[t])/(poisson(lam).pmf(z[t])*binom(x_data[t-1],alphas[0]).pmf(y[t,0]))
        prob *= binom(zprop[t],U).pmf(S[t,0])
        prob *= 1/(multinom(y[t,0],gamma).pmf(S[t,]))
        for j in range(0,q):
            prob *= binom(zprop[t-j-1],betas[j]).pmf(vprop[t,j])/binom(z[t - j - 1],betas[j]).pmf(v[t,j])
            prob *=binom(vprop[t,j],U).pmf(S[t,j+1])

    f=prob*(n**(0.5))

    J=1/(1-U)  

    A=f*J
    
    if np.random.rand(1) <= A:
        p=0
        lam = lamprop
        y=0
        alphas = 0
    return p, alphas,lam, y

def ma_to_arma(x_data, y, v,z,alphas, betas, lam,p, q,rmax)   :
    n = len(x_data)

    U=np.random.rand(1)
    alphaprop=U
    lamprop = lam*(1-U)

    S=np.zeros([n,q+1])

    for t in range(0,n):
        for j in range(1,q+1):

            S[t,j]=binom(v[t,j-1],U).rvs(1)
        S[t,1]=binom(z[t],U).rvs(1)


    vprop = np.zeros([n,q])
    yprop = np.zeros([n,1])
    zprop = np.zeros([n,1])

    for t in range(0,n):
        yprop[t]=sum(S[t,])
        zprop[t]=z[t] - S[t,0]
        for j in range(0,q):
            vprop[t,j]=v[t,j] - S[t,j-1]

    prob=1
    gamma = np.ones(q+1)

    for i in range(1,q+1) :
        gamma[i] = betas[i-1]/(sum(betas)+1)
    gamma[0]=1/(sum(betas)+1)

    for t in range(rmax-1,n):
        prob *= poisson(lamprop).pmf(zprop[t])*binom(x_data[t-1],alphaprop).pmf(yprop[t])/poisson(lam).pmf(z[t])
        prob *=multinom(yprop[t],gamma).pmf(S[t,])
        for j in range(0,q):
            prob *= binom(zprop[t-j-1],betas[j]).pmf(vprop[t,j])/ binom(zprop[t-j-1],betas[j]).pmf(v[t,j])
            prob *=1/binom(z[t],U).pmf(S[t,0])

    f=prob*n**(-0.5)

    J=(1-U)

    A=prob*J
    if np.random.rand(1) <= A:
        p=1
        alphas=alphaprop
        y=yprop
        v=vprop
        z=zprop
    return(p,alphas,y,v,z)
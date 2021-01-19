import numpy as np
from scipy.stats import poisson
from scipy.stats import binom
from scipy.stats import beta
from scipy.stats import gamma
from scipy.stats import multinomial as multinom

from INARMA_backend import *

def inarma_rjmcmc(x_data,init_augs,init_pars,init_order,order_max,N_reps):
    
    n=len(x_data)
    p_max = order_max[0]
    q_max = order_max[1]
    rmax=max(p_max,q_max)+2

    order_count= np.zeros([N_reps,2],dtype= int)
    pars_sample= []

    p = init_order[0]
    q=init_order[1]
    y = init_augs[0]
    v = init_augs[1]
    z = init_augs[2]
    alphas = init_pars[0]
    betas = init_pars[1]
    lam = init_pars[2]

    for iter in range(0,N_reps):
   

     #P-order step (ar->ar)
        if p>1:
        #Does p go up or down?
            if not p == p_max:
                order=np.random.choice([-1,1])
            else:
                order = -1

            if order==1 :
                p_new, alphas, y = p_up(x_data,y,alphas,p,rmax)
            elif order== -1 :
                p_new, alphas, y = p_down(x_data,y,alphas,p,rmax)

        elif p ==1:
                #Does p go up or down?
            while True:
                order=np.random.choice([-1,1])
                if not (order == -1 and q == 0):
                    break
            if order== -1:
                p_new,alphas, lam,y = arma_to_ma(x_data, y,v,z, alphas, betas, lam, p, q,rmax)

            if order== 1 and (not p == p_max) :
                p_new, alphas, y = p_up(x_data,y,alphas,p,rmax)
        elif p == 0 and (not q ==0) and (not p == p_max):

                p_new,alphas,y,v,z = ma_to_arma(x_data, y,v,z, alphas, betas, lam, p, q,rmax)
                
        p=p_new
       ##Q order step

        ## Gibbs-sampler for parameters
        if q==0:
            theta=sim_pars_AR(x_data,y,z,p)
            alphas=theta[0]
            lam=theta[1]
            sim_aug_AR(x_data,y,z,alphas,lam,p,rmax)
   
            pars_sample.append([alphas,[],lam])
        if p==0:
            theta=sim_pars_MA(x_data,v,z,q)
            betas=theta[0]
            lam=theta[1]
            sim_aug_MA(x_data,v,z,betas,lam,q,rmax)

            pars_sample.append([[],[betas],lam])
        if (not p ==0) and (not q ==0):
            theta=sim_pars_ARMA(x_data,y,v,z,p,q)
            alphas=theta[0]
            betas = theta[1]
            lam=theta[2]
            sim_aug_ARMA(x_data,y,v,z,alphas,betas,lam,p,q,rmax)

            pars_sample.append([[alphas],[betas],lam])


        if (iter+1) % 100 == 0:
            print("Replication : ", (iter+1))

        order_count[iter,]=[p,q]
        
    order_count=order_count.astype(int)
    samp_max_p=int(np.max(order_count[:,0]))
    samp_max_q=int(np.max(order_count[:,1]))
    par_mat=np.zeros([N_reps,samp_max_p+samp_max_q+1])
    for i in range(0, N_reps):

        alphas=pars_sample[i][0]
        betas=pars_sample[i][1]
        lam=pars_sample[i][2]
        if not len(alphas)== 0:
            par_mat[i,0:order_count[i,0]]=alphas[0]

        if not len(betas)== 0:
            par_mat[i,(samp_max_p):(samp_max_p+order_count[i,1])]=betas[0]

        par_mat[i,samp_max_p+samp_max_q]=lam
    alphas = par_mat[:,0:samp_max_p]
    betas = par_mat[:,samp_max_p:(samp_max_p+samp_max_q)]
    lams = par_mat[:,samp_max_p + samp_max_q]

    return order_count, alphas, betas, lams
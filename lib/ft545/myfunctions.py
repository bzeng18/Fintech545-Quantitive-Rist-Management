import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.stats import norm
from scipy import stats
from scipy.optimize import minimize 
###1. Covariance estimation techniques
def var(cov):
    return np.diag(cov)
def cor(cov):
    return np.diag(np.reciprocal(np.sqrt(var(cov)))) @ cov @ np.diag(np.reciprocal(np.sqrt(var(cov)))).T
def cov(var, cor):
    std = np.sqrt(var)
    return np.diag(std) @ cor @ np.diag(std).T
def expo_weighted_cov(data,lam):
    weight = np.zeros(60)
    t = len(data)
    for i in range(t):
        weight[t-1-i]  = (1-lam)*lam**i
    weight = weight/sum(weight)
    norm_data = data - data.mean()
    return norm_data.T @ (np.diag(weight) @ norm_data)
###2. Non PSD fixes for correlation matrix
def chol_psd(a):
    a = np.array(a)
    l = np.zeros_like(a)
    for i in range(len(a)):
        for j in range(i,len(a)): 
            if i==j:
                l[j,i] = a[j,i]-sum([l[i,k]**2 for k in range(j)])
                if abs(l[j][i]) <= 1e-8:
                    l[j,i] = 0
                else:
                    l[j,i] = l[j,i]**0.5
            else:
                if l[i,i] == 0:
                    l[j,i] =0
                else:
                    l[j,i] = (a[j,i]-np.dot(l[j,0:i],l[i,0:i]))/l[i,i]
    return l
def near_psd(a, epsilon=0.0):
    np.array(a)
    cov = False
    for i in np.diag(a):
        if abs(i-1)>1e-8:
            cov = True
    if cov:
        invSD = np.diag(np.reciprocal(np.sqrt(np.diag(a))))
        a = invSD @ a @ invSD
    vals, vecs = np.linalg.eigh(a)
    vals = np.array([max(i,epsilon) for i in vals])
    T = np.reciprocal(np.square(vecs) @ vals)
    T = np.diag(np.sqrt(T))
    l = np.diag(np.sqrt(vals))
    B = T @ vecs @l
    out = B @ B.T
    
    if cov:
        invSD = np.diag(np.reciprocal(np.diag(invSD)))
        out = invSD @ out @ invSD
    return out
def Ps(a, w):
    a = np.sqrt(w)@ a @np.sqrt(w)
    vals, vecs = np.linalg.eigh(a)
    vals = np.array([max(i,0) for i in vals])
    return np.sqrt(w)@ vecs @ np.diagflat(vals) @ vecs.T @ np.sqrt(w)
def Pu(a):
    b = a.copy()
    for i in range(len(a)):
        for j in range(len(a[0])):
            if i==j:
                b[i][i]=1
    return b
def F(y,a):
    d = y-a
    s = 0
    for i in range(len(d)):
        for j in range(len(d)):
            s+=d[i][j]**2
    return s
def Higham(a,w,max_iter = 1000, tor = 1e-8):
    r1 = float("inf")
    y = a
    s = np.zeros_like(y)
    for i in range(max_iter):
        r = y - s
        x = Ps(r, w)
        s = x-r
        y = Pu(x)
        r = F(y,a)
        if abs(r-r1)<tor:
            break
        else:
            r1 = r
    return y
###3.simulation method
def multi_normal_sim(cov,sim):
    return chol_psd(cov) @ np.random.normal(size = (len(cov),sim))
def simulate_pca(a, nsim, percentage = 1-1e-8):
    vals, vecs = np.linalg.eigh(a)
    tv = sum(vals)
    for i in range(len(vals)):
        i = len(vals)-i-1
        if vals[i]<0:
            vals = vals[i+1:]
            vecs = vecs[:,i+1:]
            break
        if sum(vals[i:])/tv>percentage:
            vals = vals[i:]
            vecs = vecs[:,i:]
            break
    B = vecs @ np.diag(np.sqrt(vals))
    r = np.random.normal(size = (len(vals),nsim))
    return (B @ r)
def sim_CBM(t,P0,var,sim):
    result = []
    start = P0
    for _ in range(sim):
        for i in range(t):
            start+=np.random.normal(0,np.sqrt(var))
        result.append(start)
        start = P0
    return (np.sum(result)/len(result), np.var(result))
def sim_ARS(t,P0,var,sim):
    result = []
    start = P0
    for _ in range(sim):
        start *= (1+np.random.normal(0,np.sqrt(var)))
        result.append(start)
        start = P0
    return (np.sum(result)/len(result), np.var(result))
def sim_GBM(t,P0,var,sim):
    result = []
    start = np.log(P0)
    for _ in range(sim):
        for i in range(t):
            start+=np.random.normal(0,np.sqrt(var))
        result.append(start)
        start = np.log(P0)
    return np.sum(result)/len(result), np.var(result)
###4.VaR calculation
def VaR(data, mean, alpha= 0.05):
    return -np.quantile(data-mean, q=alpha)
def return_calculate(price, method="DISCRETE"):
    price = price.pct_change().dropna()
    if method == "DISCRETE":
        return price 
    elif method == "LOG":
        return np.log(price)
def HisVaR(p, dp, alpha=5):
    portA_sim = []
    for j in range(len(p['Stock'])):
        old_value = 0
        value = 0
        index = j
        for i in p['Stock']:
            cp = dp.loc[:,i].iloc[-1]
            returns = return_calculate(dp.loc[:,i])
            value+=cp*(1+returns[index+1])*p[p['Stock']==i]['Holding'].iloc[0]
            old_value+=cp*p[p['Stock']==i]['Holding'].iloc[0]
        portA_sim.append(value)
    portA_sim -= old_value
    return -np.percentile(portA_sim,alpha)
def NormalVaR(dist, p, dp, alpha=5):
    portA_sim = []
    for j in range(len(p['Stock'])):
        old_value = 0
        value = 0
        for i in p['Stock']:
            cp = dp.loc[:,i].iloc[-1]
            returns = return_calculate(dp.loc[:,i])
            if dist == "Normal":
                rt = np.random.normal(0,np.std(returns))
            if dist == "T":
                def MLE_T(p):
                    return -1*np.sum(stats.t.logpdf(returns, df=p[0], loc = p[1],scale=p[2])) 
                constraints=({"type":"ineq", "fun":lambda x: x[0]-1}, 
                 {"type":"ineq", "fun":lambda x: x[2]})
                df, loc, scale = minimize(MLE_T, x0 = (10,np.mean(returns),np.std(returns)),constraints=constraints).x
                rt= stats.t(df=df, scale=scale).rvs(1)
            value+=cp*(1+rt)*p[p['Stock']==i]['Holding'].iloc[0]
            old_value+=cp*p[p['Stock']==i]['Holding'].iloc[0]
        portA_sim.append(value)
    portA_sim -= old_value
    return -np.percentile(portA_sim,alpha)
###5. ES
def es(data, alpha = 0.05):
    data = list(data)
    sorted = np.sort(data)
    index = round(alpha*len(sorted))
    return -np.mean(sorted[:index])
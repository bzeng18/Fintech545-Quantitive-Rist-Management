import pandas as pd
import numpy as np
from ft545 import myfunctions
def test_expo_weight_cov():
    data = pd.read_csv("tests/DailyReturn.csv")
    data = data.drop("Unnamed: 0",axis=1)
    assert sum(myfunctions.expo_weighted_cov(data,0.5)["SPY"])==0.004514464371270292
def test_psd():
    n = 500
    sigma = np.full((n,n),0.9)
    for i in range(n):
        sigma[i,i]=1.0
    sigma[0,1] = 0.7357
    sigma[1,0] = 0.7357
    assert np.all(np.linalg.eigvals(myfunctions.near_psd(sigma)))>-1e-8
    assert np.all(np.linalg.eigvals(myfunctions.Higham(sigma, np.identity(len(sigma)))))>-1e-8
def test_covsim():
    data = pd.read_csv("tests/DailyReturn.csv")
    data = data.drop("Unnamed: 0",axis=1)
    norm_cov = np.cov(data.T)
    new_cov1 = myfunctions.cov(np.var(data), myfunctions.cor(norm_cov))
    assert sum(new_cov1[0])==0.007185171936793685
    data1 = myfunctions.multi_normal_sim(new_cov1,25000)
    data2 = myfunctions.simulate_pca(new_cov1, 25000, percentage = 1)
    assert myfunctions.F(new_cov1,np.cov(data1))<1e-7
    assert myfunctions.F(new_cov1,np.cov(data2))<1e-7
def test_sim():
    t = 100
    P0= 100
    var = 0.01
    sim = 10000
    assert abs(myfunctions.sim_CBM(t,P0,var,sim)[0]-P0)<0.1
    assert abs(myfunctions.sim_CBM(t,P0,var,sim)[1]-t*var)<0.1
    assert abs(myfunctions.sim_ARS(t,P0,var,sim)[0]-P0)<0.3
    assert abs(myfunctions.sim_ARS(t,P0,var,sim)[1]-P0**2*var)<3
    assert abs(myfunctions.sim_GBM(t,P0,var,sim)[0]-np.log(P0))<0.1
    assert abs(myfunctions.sim_GBM(t,P0,var,sim)[1]-t*var)<0.1
def test_HistVaR():
    p = pd.read_csv("tests/portfolio.csv")
    dp = pd.read_csv("tests/DailyPrices.csv")
    assert myfunctions.HisVaR(p[p["Portfolio"]=='A'],dp)==4885.46827979658
def test_VaR():
    data = pd.read_csv("tests/INTC.csv")
    returns = myfunctions.return_calculate(data['INTC'])
    assert myfunctions.VaR(returns, np.mean(returns)) == 0.029574903865632305
def test_es():
    data = pd.read_csv("tests/INTC.csv")
    returns = myfunctions.return_calculate(data['INTC'])
    assert myfunctions.es(returns) == 0.057392569738886345

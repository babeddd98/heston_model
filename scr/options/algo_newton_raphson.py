from black_sholes import bs_call

def vol_implicite(S, K, T, r, mkt_price, n=5):
    x = 0
    y = 1
    for i in range(1,n):
        z = 0.5*(x + y)
        if mkt_price > bs_call(S,K,0,T,r,(z/(1-z))):
            x = z
        else:
            y = z
    z = 0.5*(x + y)
    sigma = z/(1-z)
    return sigma
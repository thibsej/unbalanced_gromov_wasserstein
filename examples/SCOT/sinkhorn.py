import numpy as np
import matplotlib.pyplot as plt
import progressbar

def solve(a,b,C,epsilon,niter=100,rho=np.Inf, verb=True):
    """ Sinkhorn's algorithm.

    Parameters
    ----------
    a : vector of length n, the first histogram

    b : vector of length m, the second histogram

    C : matrix of size (n,m), the cost matrix

    epsilon : real>0, the regularization parameter

    niter : integer, default=100, the number of iteration

    rho : real>0, default=np.Inf, the unbalanced relaxation of the marginals
        Set rho=np.Inf for classical (balanced) OT


    Returns
    -------
    P : matrix of size (n,m), the transport plan, with marginal a and b
        P_ij = a_i exp((f_i+g_j-C_ij)/epsilon) * b_j
    f : vector of length n, the firt dual potential

    g : vector of length m, the second dual potential

    Err : L1 norm between a and np.sum(P,axis=1) during the iterations.

    Example
    -------
    >>> import sinkhorn
    >>> a,b = np.random.rand(n), np.random.rand(m)
    >>> a,b = a/np.sum(a), b/np.sum(b)
    >>> X,Y = np.random.randn(n,2), np.random.randn(m,2)
    >>> C = np.sum( (X[:,None,:]-Y[None,:,:])**2, axis=2 )
    >>> P,f,g,Err = sinkhorn.solve(a,b,C,epsilon=.1,niter=5000)
    >>> plt.plot(a, 'r')
    >>> plt.plot(np.sum(P,axis=1), 'r.')
    """

    # stabilized c transform
    def mina_u(H,epsilon): return -epsilon*np.log( np.sum(a[:,None] * np.exp(-H/epsilon),0) )
    def minb_u(H,epsilon): return -epsilon*np.log( np.sum(b[None,:] * np.exp(-H/epsilon),1) )
    def mina(H,epsilon): return mina_u(H-np.min(H,0),epsilon) + np.min(H,0);
    def minb(H,epsilon): return minb_u(H-np.min(H,1)[:,None],epsilon) + np.min(H,1);
    n = len(a)
    m = len(b)
    kappa = rho/(rho+epsilon)
    if rho==np.Inf:
        # balanced OT
        kappa = 1
    f = np.zeros(n)
    Err = np.zeros(niter)
    R = range(niter)
    if verb==True:
        R = progressbar.progressbar(R)
    for it in R:
        g = kappa*mina(C-f[:,None],epsilon)
        if rho==np.Inf:
            P = a[:,None] * np.exp((f[:,None]+g[None,:]-C)/epsilon) * b[None,:]
            a1 = np.sum(P,axis=1)
            Err[it] = np.linalg.norm(a-a1, 1)
        f = kappa*minb(C-g[None,:],epsilon)
    # generate the coupling
    P = a[:,None] * np.exp((f[:,None]+g[None,:]-C)/epsilon) * b[None,:]
    return P,f,g,Err

def test_points():

    n = 20
    m = 30

    a,b = np.random.rand(n), np.random.rand(m)
    a,b = a/np.sum(a), b/np.sum(b)
    X,Y = np.random.randn(n,2), np.random.randn(m,2)
    C = np.sum( (X[:,None,:]-Y[None,:,:])**2, axis=2 )

    epsilon = .1
    niter = 5000
    P,f,g,Err = solve(a,b,C,epsilon,niter=niter)

    plt.subplot(2,2,1)
    plt.plot(np.log(Err))
    plt.subplot(2,2,2)
    plt.imshow(P)
    plt.subplot(2,2,3)
    plt.plot(a, 'r')
    plt.plot(np.sum(P,axis=1), 'r.')
    plt.subplot(2,2,4)
    plt.plot(b, 'b')
    plt.plot(np.sum(P,axis=0), 'b.')

    return P,f,g,Err, X,Y,a,b

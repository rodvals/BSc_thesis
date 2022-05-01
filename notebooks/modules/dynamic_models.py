import numpy as np
import scipy
from scipy.integrate import odeint
import igraph as ig


def rossler(_q, _t, _a, _b, _c, _r, _sigma, _N, _adj):
    _dq = np.zeros(np.shape(_q))
    x, y, z = _q[0::3], _q[1::3], _q[2::3]
    dx, dy, dz = _dq[0::3], _dq[1::3], _dq[2::3]
    
    # N nodi della rete
    for i in range(_N):
        dx[i] =  -y[i] - z[i] + _sigma/(2*_r*_N) * np.sum( _adj[i] * (x - x[i]) )
        dy[i] =  x[i] + _a*y[i] + _sigma/(2*_r*_N) * np.sum( _adj[i] * (y - y[i]) )
        dz[i] =  _b + z[i]*(x[i] - _c) + _sigma/(2*_r*_N) * np.sum( _adj[i] * (z - z[i]) )
            
    return _dq
    
def kuramoto(_q,_t,  _omegas, _K, _N, adj):
    _dq = np.zeros(np.shape(_q))
    for i in range(_N):
        for j in range(_N):
            _dq[i] += adj[i][j] * np.sin(_q[j] - _q[i])
        _dq[i] /=  ( np.sum(adj[i]) + 1 )
    _dq *= _K
    _dq += _omegas
    return _dq

def r_parameter(_q, _N):
    _r = np.abs( np.sum( np.exp(np.cdouble(_q)*1j) / _N ))
    return _r

def r_mean_parameter(_q_0, _t_span, _omegas, _K , _N, _adj):
    _r_mean = 0
    _q, _dq = motion( _q_0, _t_span, _omegas, _K, _N, _adj)
    for t_count in range(len(_t_span)):
        _r_mean += r_parameter(_q[t_count,:], _N)
    #media
    _r_mean *= 1./len(t_span)
    return _r_mean

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from numpy import exp
import numpy as np
from scipy.stats import binom

"""
Function ExpectedUB
Calculates expected benefits with expected returns utility function
Input: t0 = the beginning of the policy 
       t2 = last chance to lapse
       t3 = early death for policy holder
       t4 = normal life expectancy
       t5 = late death for policy holder
       q3, q4 = probability of dying at t_i step
       dt = payment interval
       alpha = parameters for the utility function
       rho = discount rate for the policyholder 
       S0 = initial payment
       q = percentage of the initial payment for the cash payout
"""
def ExpectedUB(alpha, beta, S0, q, t, r, q3, q4, dt, rho):
    return (alpha*q*S0+beta)*(1-exp(-rho*(t[5]-t[2]))-(exp(-rho*(t[3]-t[2]))-exp(-rho*(t[5]-t[2])))*q3
                       -(exp(-rho*(t[4]-t[2]))-exp(-rho*(t[5]-t[2])))*q4)/(exp(dt)-1)
    
def Boundary1(alpha, beta, r, lambd, sigma, dt, N, K, rho, t1, t2):
    B = []
    # Case 1
    tmp = K*exp(-rho*(t2-t1))-beta
    if 0 <= tmp and tmp < (K-beta)*exp(-N*sigma*np.sqrt(dt)):
        B.append([tmp/alpha, np.inf])

    # Case 2
    tmp = beta*(exp(-rho*(t2-t1))-1)/(1-exp(r+lambd*sigma-rho)*(t2-t1))
    if rho<r+lambd*sigma and (K-beta)*exp(N*sigma*np.sqrt(dt))<=tmp:
        # is the second part right?
        B.append([beta*(exp(-rho(t2-t1))-1)/(1-exp(r+lambd*sigma-rho)(t2-t1)), np.inf])
    
    p = ((exp((r+lambd*sigma)*dt))-exp(-sigma*np.sqrt(dt)))/(exp(sigma*np.sqrt(dt))-exp(-sigma*np.sqrt(dt)))
    # i think this is the range
    for j in range(1, N+1):
        P = binom.cdf(j-1,N,p)
        # i've found another formula on the paper 
        S = ((exp(-rho*(t2-t1))*P-1)*beta+K*(1-P)*exp(-rho*(t2-t1)))/alpha/(1-P*exp(r+lambd*sigma-rho)*(t2-t1))
        if S <= (K-beta)*exp((N-2*j+2)*sigma*np.sqrt(dt))/alpha:
            # minor fix on the boundary
            U = (K-beta)*exp((N-2*j+2)*sigma*np.sqrt(dt))/alpha
            L = max(S,(K-beta)*exp((N-2*j)*sigma*np.sqrt(dt))/alpha)
            #B=np.append(B,[L,U])
            B.append([L,U])
 
    # Now we check if any intervals overlap and condense the list
    B=B[::-1]
    lapseIntervals = [B[0]]
    for i in range(1, len(B)):
        # If the lowerbound of the next interval is within the previous interval and the upperbound is greater than the previous interval, merge
        if lapseIntervals[-1][1] >= B[i][0]:
            if lapseIntervals[-1][1] < B[i][1]:
               lapseIntervals[-1][1] = B[i][1]
        else:
            lapseIntervals.append(B[i])

    return np.array(lapseIntervals)

""" Function LapseBoundary
    Input: r - interest forward rate 
    lambd - customer preference forward rate
    sigma - variance of equity 
    dt - distance between time points 
    alpha - parameter of utility funcion 
    N - Total number of upward movement of equity 
    K - expectedUB 
    rho - forward rate 
    t - array of time steps in years
    Q - array of probabilities of dying starting at t3
    Returns: A function taking in a timestep t relative to the start of the policy in years, and outputs the predicted value of S.
             If the actual S value is higher than the predicted value, the policy holder will lapse
""" 
def LapseBoundary(S0, r, lambd, sigma, dt, alpha, beta, N, rho, t, q, Q):
    K = ExpectedUB(r=r, t=t, q3=Q[0], q4=Q[1], beta=beta, alpha=alpha, S0=S0, q=q, rho=rho, dt=dt)
    # Boundary at t1
    t1Boundary = Boundary1(alpha, beta, r, lambd, sigma, dt, N, K, rho, t[1], t[2])
    # Boundary at t2
    t2Boundary = (K-beta)/alpha
    return [t1Boundary, t2Boundary]

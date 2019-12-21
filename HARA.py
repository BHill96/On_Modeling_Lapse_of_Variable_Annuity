#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from numpy import exp
import numpy as np
from scipy.stats import binom

# Parameters Representing: 
    # t0 = the beginning of the policy
    # t2 = the last chance for the policyholder to early withdraw 
    # t3 = polictholders pass away at t3 with probability q3
    # t4 = polictholders pass away at t4 with probability q4
    # t5 = polictholders pass away at t5 with probability 1-q3-q4
    # q3, q4 = see above
    # deltaT = payment interval
    # alpha = parameters for the utility function
    # rho = discount rate for the policyholder 
    # S0 = initial payment
    # q = percentage of the initial payment the policyholder gets every period
# Returns the expected utility at t2 of the forecoming payments
def ExpectedUB(t0, t2, t3, t4, t5, q3, q4, deltaT, alpha, rho, S0, q):
  # see 2.24 in paper
  return ((q*S0)**alpha)*(1-exp(-rho*(t5-t2))-(exp(-rho*(t3-t2))-exp(-rho*(t5-t2)))*q3
                       -(exp(-rho*(t4-t2))-exp(-rho*(t5-t2)))*q4)/(exp(deltaT)-1)
  
""" Function Boundary1
    Input: r - forward rate 
    lambd - forward rate
    sigma - variance of equity 
    dt - distance between time points 
    alpha - parameter of utility funcion 
    N - Total number of upward movement of equity 
    K - expectedUB 
    rho - forward rate 
    t1 - time at first chance to lapse 
    t2 - time at last chance to lapse 
    Returns: List containing intervals in which the policy holder will lapse is S exists inside those intervals 
""" 
def Boundary1(r, lambd, sigma, dt, alpha, N, K, rho, t1, t2): 
    B = []
    p = ((exp((r+lambd*sigma)*dt))-exp(-sigma*np.sqrt(dt)))/(exp(sigma*np.sqrt(dt))-exp(-sigma*np.sqrt(dt)))
    # If S>K
    if exp(-rho*(t2-t1))*(p*exp(alpha*sigma*np.sqrt(dt)+(1-p)*exp(-alpha*sigma*np.sqrt(dt))))**N < 1:
        B.append([exp(N*sigma*np.sqrt(dt))*(K**(1/alpha)), np.inf])

    # If K >= S
    if rho*(t2-t1)/alpha >= (N+2)*sigma*np.sqrt(dt):
        B.append([(exp(-rho*(t2-t1))*K)**(1/alpha), (exp(-sigma*alpha*np.sqrt(dt)*N)*K)**(1/alpha)])
    
    # i think this is the range
    for j in range(1, N+1):
        P = binom.cdf(j-1,N,p)
        # i've found another formula on the paper 
        S = ((K*P)/(exp(rho*(t2-t1))-(1-P)*(p*exp(alpha*sigma*np.sqrt(dt))+(1-p)*exp(-alpha*sigma*np.sqrt(dt)))**N))**(1/alpha)
        if S <= (K**(1/alpha))*exp(-(2*j-2-N)*sigma*np.sqrt(dt)):
            # minor fix on the boundary
            U = K**(1/alpha)*exp(-(2*j-2-N)*sigma*np.sqrt(dt))
            L = max(S,K**(1/alpha)*exp(-(2*j-N)*sigma*np.sqrt(dt)))
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
def LapseBoundary(S0, r, lambd, sigma, dt, alpha, N, rho, t, q, Q):
    K = ExpectedUB(t[0], t[2], t[3], t[4], t[5], Q[0], Q[1], dt, alpha, rho, S0, q)
    # Boundary at t1
    t1Boundary = Boundary1(r, lambd, sigma, dt, alpha, N, K, rho, t[1], t[2])
    # Boundary at t2
    t2Boundary = K**(1/alpha)
    return [t1Boundary[0][0], t2Boundary]

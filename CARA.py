#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from numpy import exp, sqrt, ceil, log
from math import factorial

 # Parameters Representing: 
    # t0 = the beginning of the policy 
    # t2 = the last chance for the policyholder to early withdraw 
    # t3 = polictholders pass away at t3 with probability q3
    # t4 = polictholders pass away at t4 with probability q4
    # t5 = polictholders pass away at t5 with probability 1-q3-q4
    # q3, q4 = see above
    # deltaT = payment interval
    # alpha, beta = parameters for the utility function
    # rho = discount rate for the policyholder 
    # S0 = initial payment
    # q = percentage of the initial payment the policyholder gets every period
# Returns the expected utility at t2 of the forecoming payments
def ExpectedUB(t0, t2, t3, t4, t5, q3, q4, deltaT, alpha, beta, rho, S0, q):
  # see 2.24 in paper
  return -alpha*(exp(-beta*q*S0))*(1-exp(-rho*(t5-t2))
                      -(exp(-rho*(t3-t2))-exp(-rho*(t5-t2)))*q3
                      -(exp(-rho*(t4-t2))-exp(-rho*(t5-t2)))*q4)/(exp(deltaT)-1)

def lambda1sol(t0, t1, t2, deltaT, alpha, beta, rho, S0, S1, sigma, p, K, n):
    # range does not include the stop number so all stop numbers had one added to them
    # Find denominator
    # This value can easily be rounded to 0. We set it to a small number if ithat happens.
    temp = exp(-beta*S1*exp(sigma*sqrt(deltaT)*n))
    if temp == 0:
        temp = 10**-8
    if -K/(alpha*temp) < 1 or -K/alpha > 1:
        # If this happens, the probability of 2.34 is 0, making R1 the expected value of the utility of S_1
        RSumBound = (t2-t1)/deltaT
        R1 = exp(-rho*(t2-t1))
        R1 *= -alpha*factorial(RSumBound)*sum([exp(-beta*S1*exp(sigma*sqrt(deltaT)*(2*k-RSumBound)))*(p**k)*(
                (1-p)**(RSumBound-k))/(factorial(k)*factorial(RSumBound-k)) for k in range(0, int(RSumBound+1))])
    else:
        sumBound = int(ceil((log(-log(-K/alpha)/(beta*S1))/(2*sigma*np.sqrt(deltaT)))+(n/2)))
        R1 = 0
        # This means the probability of the utility of S_1 being greater than the expected value of UB is not 0
        if sumBound >= 0:
            RSumBound = (t2-t1)/deltaT
            R1 *= -alpha*factorial(RSumBound)*sum([exp(-beta*S1*exp(sigma*sqrt(deltaT)*(2*k-RSumBound)))*(p**k)*(
                    (1-p)**(RSumBound-k))/(factorial(k)*factorial(RSumBound-k)) for k in range(0, int(RSumBound+1))])
            R1 *= sum([factorial(n)*(p**j)*((1-p)**(n-j))/(factorial(j)*factorial(n-j)) for j in range(0, sumBound)])
            R1 += K*sum([factorial(n)*(p**j)*((1-p)**(n-j))/(factorial(j)*factorial(n-j)) for j in range(sumBound, n+1)])
            R1 *= exp(-rho*(t2-t1))
            
        # We define factorials of negative numbers to be 0
        if sumBound < 0:
            sumBound = 0
            R1 += K

    # Find numerator
    L1 = -alpha*exp(-beta*S1)
    temp = []
    temp.append(L1)
    temp.append(R1)
    return temp

def lambda1draw(t, deltaT, alpha, beta, rho, S0, sigma, n, Q, q) :
  K = ExpectedUB(t[0], t[2], t[3], t[4], t[5], Q[0], Q[1], deltaT=deltaT, alpha=alpha, beta=beta, rho=rho, S0=S0, q=q)
  u = np.exp(sigma*np.sqrt(deltaT))
  d = np.exp(-sigma*np.sqrt(deltaT))
  # Risk probability 
  a = np.exp(0.02*1/12+0.1*0.05*1/12)
  p = (a-d)/(u-d)
  x = list(range(0, 2001))
  x = [35372.677547714484+a*(282704.0725574398-35372.677547714484)/2000 for a in x]
  y1 = []
  y2 = []
  for S1 in x:
    temp = lambda1sol(t[0], t[1], t[2], deltaT, alpha, beta, rho, S0, S1, sigma, p, K, n)
    y1.append(temp[0])
    y2.append(temp[1])
  
  plt.scatter(x, y1, label='y1')
  plt.scatter(x, y2, label='y2')
  plt.legend()
  plt.title('CARA')
  plt.show()
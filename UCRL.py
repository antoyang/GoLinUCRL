# =============================================================================
# updates
# change transition function to operate directly on string and avoid calling ast.literal_eval
# =============================================================================

import numpy as np
#from itertools import product as itp
from copy import deepcopy
#import ast
from random import shuffle
import time
import pandas as pd
import re

def indexes(l,elt):
    indices = [i for i, x in enumerate(l) if x == elt]
    return indices

def decay(delta):
    if delta==0:
        out = 0.
    else:
        out = 1./delta
    return out

def state_unstr(str_state):
    #state = [int(s) for s in re.findall(r'\d+', str_state)] # little bit slower
    #state = [int(elt) for elt in str_state if elt.isdigit()] # fastest but works only if all elt are < 10
    state = list(map(int,re.findall(r'\d+', str_state)))
    return state

def ucrl_conf_weight_r(x,conf):
    if x>0:
        out = conf/(2*x)
        out = min(1,np.sqrt(conf))
    else:
        out = 1
    return out

# UCRL in the DMDP case
class UCRL(object):
    
    def __init__(self,alpha,states,s0,actions,reward,transition):
        """
        Parameters
        =========
        alpha: float
            exploration parameter
        states: list of list
            list of all states. Each state is a list in str form eg: '[1, 1, 0]'
        s0: list in str form
            initial state
        actions: list
            list of possible actions
        reward: function
            function of state and action that return the reward
        transition: function
            function of state and action that return the next state
        """
        
        self.alpha = alpha
        self.states = states
        self.actions = actions
        self.cardS = len(states)
        self.cardA = len(actions)
        self.round = 0
        self.reward = reward # true reward function
        self.transition = transition # deterministic transition
        self.policy = dict.fromkeys(states)
        self.conf = pd.DataFrame(columns=actions,index=states).fillna(0)
        self.t = 0
        self.reward_sequence = []
        self.arm_sequence = []
        
        # count (action,state) occurences
        self.N = pd.DataFrame(columns=actions,index=states).fillna(0)
        # sum (action,state) rewards
        self.R = pd.DataFrame(columns=actions,index=states).fillna(0)
        # rewards upper bound
        self.r_hat = pd.DataFrame(columns=actions,index=states).fillna(0)
        for s in states:
            self.policy[s] = np.random.randint(0,self.cardA)

        # choose first state at random
        self.state = s0
    
    def Bellman_a(self,a,V,gamma,str_state,rewards):
        """
        Bellman update for a particular action,state
        """
        # update for one state
        r = rewards[str_state][a]+self.conf.loc[str_state,a] # UCB part
        next_state = self.transition(str_state,a)
        r+=gamma*V[next_state]
        return r
        
    def policyUpdate(self,rewards):
        """
        Compute policy based on current rewards estimates using Value Iteration
        """
        #t0 = time.time()
        V = dict.fromkeys(self.states)
        for str_state in self.states:
            V[str_state] = 0
        
        # Value Iteration
        for it in range(self.n_iter):
            shuffle(self.states)
            for str_state in self.states:
                V[str_state] = max(list(map(lambda a: self.Bellman_a(a,V,self.gamma,str_state,rewards),range(self.cardA))))
        
        # Compute policy using value function
        for str_state in self.states:
            tmp = np.zeros(self.cardA)
            for a in range(self.cardA):
                r = rewards[str_state][a]+self.conf.loc[str_state,a] # UCB
                next_state = self.transition(str_state,a)
                r+=self.gamma*V[next_state]
                tmp[a] = r
            self.policy[str_state] = np.argmax(tmp)
#        t1 = time.time()
#        tot = t1-t0
#        print("Policy update time %f seconds" % tot)

    def run(self,T=1000,gamma=1,n_iter=100,verbose=True):
        """
        Parameters
        ========
        T: int
            number of iterations of UCRL
        gamma: float
            discount parameter of Bellman operator
        n_iter:
            number of iterations of Bellman operator
        """
        self.gamma = gamma
        self.update_time = []
        self.n_iter = n_iter
        while self.t<T:
            if (verbose) & (self.t%10==0):
                print("iteration",self.t)
            # choose next action
            a = self.policy[self.state]
            self.arm_sequence.append(a)
            # Observe reward
            r = self.reward(state_unstr(self.state),a)
            self.reward_sequence.append(r)
            # update      
            self.N.loc[self.state,a]+=1
            self.R.loc[self.state,a]+=r
            self.state = self.transition(self.state,a) # next state
            update_policy = False
            _conf = np.log(2*(self.t+1**self.alpha)*self.cardS*self.cardA)
            new_conf = self.N.applymap(lambda x: ucrl_conf_weight_r(x,_conf))
            if (new_conf<self.conf/2).max().max():
                update_policy = True
            self.r_hat = self.R/self.N
            self.r_hat = self.r_hat.fillna(0)
            
            # update policy ?
            if update_policy:
                self.round+=1
                self.update_time.append(self.t+1)
                if verbose:
                    print('update policy, round=',self.round)
                self.conf = deepcopy(new_conf)
                self.policyUpdate(self.r_hat)
            
            self.t+=1

#==============================================================================
class linUCRL(object):
    
    def __init__(self,alpha,d,states,s0,actions,reward,transition):
        """
        Parameters
        =========
        alpha: float
            exploration parameter
        d: int
            degree of the linear part
        states: list of list
            list of all states. Each state is a list in str form eg: '[1, 1, 0]'
        s0: list in str form
            initial state
        actions: list
            list of possible actions
        reward: function
            function of state and action that return the reward
        transition: function
            function of state and action that return the next state
        """
        
        self.alpha = alpha
        self.d = d # dimension -1
        l_w = 1+np.log(len(states[0]))
        self.L = np.sqrt((1-l_w**(d+1))/(1-l_w))
        self.states = states
        self.actions = actions
        self.cardS = len(states)
        self.cardA = len(actions)
        self.round = 0
        self.reward = reward # true reward function
        self.transition = transition # deterministic transition
        self.policy = dict.fromkeys(states)
        self.phi = dict.fromkeys(actions) # reward parameters
        self.conf = dict.fromkeys(actions)
        self.N = dict.fromkeys(actions) # time spend in a since begin
        self.roundN = dict.fromkeys(actions) # time spend in a in current round
        self.R = dict.fromkeys(actions) # reward sequence for each arm
        self.X = dict.fromkeys(actions) # observed context
        self.t = 0
        self.reward_sequence = []
        self.arm_sequence = []
        
        for s in states:
            self.policy[s] = np.random.randint(0,self.cardA)

        for a in actions:
            self.R[a] = []
            self.X[a] = []
            self.N[a] = 0
            self.roundN[a] = 0
            self.conf[a] = 1
            self.phi[a] = np.array([1.]+list(np.zeros(self.d))).reshape(1,self.d+1)

        # choose first state at random
        self.state = s0
    
    def conf_r(self,t,a,lbda):

        conf = np.log(self.cardA*t**self.alpha*(1+self.N[a]*self.L**2/lbda))
        conf = np.sqrt((self.d+1)*conf)+np.sqrt(lbda)
        if self.N[a]==0:
            conf*=10.
        else:
            conf*=0.01
        #conf = min(1,conf)
        return conf
    
    def Bellman_a(self,a,V,gamma,str_state,rewards):
        """
        Bellman update for a particular action,state
        """
        # update for one state
        r = rewards[str_state][a]
        next_str_state = self.transition(str_state,a)
        r+=gamma*V[next_str_state]
        return r
        
    def policyUpdate(self):
        """
        Compute policy based on current rewards estimates using Value Iteration
        """
#        t0 = time.time()
        
        V = dict.fromkeys(self.states)
        for state in self.states:
            V[state] = 0
        
        # compute rewards upper bound
        _rewards = dict.fromkeys(self.states)
        for str_state in self.states:
            _rewards[str_state] = dict.fromkeys(list(range(self.cardA)))
            for a in range(self.cardA):
                state = state_unstr(str_state)
                tmp = sum([decay(delta+1) for delta in indexes(state,a)]) # compute decay
                x = np.array([tmp**j for j in range(self.d+1)]).reshape(1,self.d+1)
                r = self.phi[a].dot(x.T)+self.conf[a]*np.sqrt(x.dot(np.linalg.solve(self.V[a],x.T))) # UCB
                r = r[0][0]
                _rewards[str_state][a] = r
        
        # apply Value Iteration
        for it in range(self.n_iter):
            shuffle(self.states)
            for str_state in self.states:
                V[str_state] = max(list(map(lambda a: self.Bellman_a(a,V,self.gamma,str_state,_rewards),range(self.cardA))))
        
        # find policy
        for str_state in self.states:
            state = state_unstr(str_state)
            r_arms = np.zeros(self.cardA)
            for a in range(self.cardA):
                tmp = sum([decay(delta+1) for delta in indexes(state,a)]) # compute decay
                x = np.array([tmp**j for j in range(self.d+1)]).reshape(1,self.d+1)
                r = self.phi[a].dot(x.T)+self.conf[a]*np.sqrt(x.dot(np.linalg.solve(self.V[a],x.T))) # UCB
                r = r[0][0]
                next_state = self.transition(str_state,a)
                r+=self.gamma*V[next_state]
                r_arms[a] = r
            self.policy[str_state] = np.argmax(r_arms)
#        t1 = time.time()
#        tot = t1-t0
#        print("Policy update time %f seconds" % tot)
    
    def run(self,T=1000,lbda=0.5,gamma=1,n_iter=100):
        """
        Parameters
        ========
        T: int
            number of iterations of UCRL
        lbda: float
            regularization for linear model learning
        gamma: float
            discount parameter of Bellman operator
        n_iter:
            number of iterations of Bellman operator
        """
        
        self.V = dict.fromkeys(self.actions)
        for a in self.actions:
            self.V[a] = lbda*np.eye(self.d+1)
        
        self.gamma = gamma
        self.update_time = []
        self.n_iter = n_iter
        while self.t<T:
            # choose next action
            if self.t<self.cardA:
                a = self.t # pull each arm at least once
            else:
                a = self.policy[self.state]
            self.arm_sequence.append(a)
            
            # Observe reward
            r = self.reward(state_unstr(self.state),a)
            self.R[a].append(r)
            self.reward_sequence.append(r)
            
            # update V, X
            state = state_unstr(self.state)
            tmp = sum([decay(delta+1) for delta in indexes(state,a)]) # compute decay
            x = np.array([tmp**j for j in range(self.d+1)]).reshape(1,self.d+1)
            self.V[a] += x.T.dot(x)
            self.X[a].append(x[0])
            
            # update
            self.roundN[a]+=1
            self.state = self.transition(self.state,a) # next state
            update_policy = False
            for a in self.actions:
                if (self.roundN[a] >= self.N[a]) & (self.roundN[a]>1):
                    update_policy = True
            
            # update policy ?
            if update_policy:
                self.round+=1
                for a in self.actions:
                    self.N[a] += self.roundN[a]
                    self.roundN[a] = 0
                    XR = np.array(self.X[a]).T.dot(np.array(self.R[a]))
                    try:
                        self.phi[a] = np.linalg.solve(self.V[a],XR).reshape(1,self.d+1)
                    except:
                        pass # arm still not pulled
                    self.conf[a] = self.conf_r(self.t+1,a,lbda)
                self.update_time.append(self.t+1)
                print('update policy, round=',self.round)
                self.policyUpdate()
            else:
                print("not update")

            self.t+=1
            print("iteration number:",self.t)
            
            
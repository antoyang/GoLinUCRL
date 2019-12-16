# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 11:54:48 2019

@author: Antoine
"""
import numpy as np
from scipy.linalg import sqrtm
import re
from random import shuffle

def indexes(l,elt):
    """
    return indices of elt in list l
    """
    indices = [i for i, x in enumerate(l) if x == elt]
    return indices

def decay(delta):
    if delta==0:
        out = 0.
    else:
        out = 1./delta
    return out

def state_unstr(str_state):
    """
    converts str_state into int state_index
    """
    state = list(map(int,re.findall(r'\d+', str_state)))
    return state

def phi(x,i,n):
    """
    Parameters
    =========
    x: vector to be place in the sparse output
    i: position of x (when divided by the length of x, true pos = length(x)*i)
    n: n*length(x) = total vector size
    """
    d = len(x)
    out = np.zeros(n*d)
    out[i*d:(i+1)*d] = x
    out = out.reshape(-1,1)
    return out

class GLUCRL(object):
    def __init__(self,alpha,d,states,actions,reward,transition,L):
        """
        Parameters
        =========
        alpha: float
            exploration parameter
        d: int
            degree of the linear part
        states: list of list
            list of all states. Each state is a list in str form eg: '[1, 1, 0]'
        actions: list
            list of possible actions
        reward: function
            function of state, action and user that return the reward
        transition: function
            function of state and action that return the next state
        L: matrix
            laplacian of the graph
        """
        
        self.alpha = alpha
        self.d = d
        self.states = states
        self.actions = actions
        self.reward = reward #true reward function
        self.transition = transition
        self.L = L
        self.nS = len(states)
        self.nA = len(actions)
        self.nUsers = L.shape[0]
        
        A = np.eye(self.nUsers)+L
        A_inv_sqrt = np.real(sqrtm(np.linalg.inv(A))) # Real to avoid numerical issues 
        self.A_kro_inv_sqrt = np.kron(A_inv_sqrt,np.eye(self.d+1))
        
        #policies for each user initialized randomly
        self.policy = dict.fromkeys(range(self.nUsers)) 
        for user in range(self.nUsers):
            self.policy[user] = dict.fromkeys(list(map(str,states)))
            for s in states:
                self.policy[user][str(s)] = np.random.randint(0,self.nA)
                
        self.theta = dict.fromkeys(actions) # reward parameter
        self.N = dict.fromkeys(actions) # time spent in a since start
        self.roundN = dict.fromkeys(actions) # time spent in a in current round
        self.R = dict.fromkeys(actions) # reward sequence for each arm 
        self.conf = dict.fromkeys(actions) # ucb parameter
        for a in actions:
            self.theta[a] = np.array([1.]+list(np.zeros((self.d+1)*self.nUsers-1))).reshape(1,(self.d+1)*self.nUsers)
            self.N[a] = 0
            self.roundN[a] = 0
            self.R[a] = []
            self.conf[a] = 1
            
        self.reward_sequence = []
        self.user_sequence = []
        self.arm_sequence = dict.fromkeys(range(self.nUsers))
        for u in range(self.nUsers):
            self.arm_sequence[u] = []
        self.t = np.zeros(self.nUsers) # time spent exploring for each user
        self.round = 0
        
        # initial states initialized randomly
        self.cur_states = dict.fromkeys(range(self.nUsers)) 
        """for u in range(self.nUsers):
            idx = np.random.choice(self.nS)
            self.cur_states[u] = self.states[idx]"""
        for u in range(self.nUsers):
            self.cur_states[u] = self.states[0]

            
    def conf_r(self,t,a,lbda):
        """
        Calculates the ucb coefficient at time t for action a given lambda
        """
        l_w = 1 + np.log(len(self.states[0]))
        L = np.sqrt((1 - l_w ** (self.d + 1)) / (1 - l_w))
        conf = np.log(self.nA*t**self.alpha*(1+self.N[a]*L**2/lbda))
        conf = np.sqrt((self.d+1)*conf)+np.sqrt(lbda)
        if self.N[a]==0:
            conf*=10.
        else:
            conf*=0.01
        return conf
    
    def Bellman_a(self,a,V,gamma,str_state,rewards):
        """
        Bellman update for a particular action,state
        """
        # update for one state
        r = rewards[str_state][a]
        next_state = self.transition(str_state,a)
        r+=gamma*V[next_state]
        return r

    def policyUpdate(self, user):
        # Initialize value function for Value Iteration
        Vf = dict.fromkeys(self.states)
        for str_state in self.states:
            Vf[str_state] = 0

        # Compute rewards upper bound
        _rewards = dict.fromkeys(self.states)
        for str_state in self.states:
            _rewards[str_state] = dict.fromkeys(list(range(self.nA)))
            for ac in range(self.nA):
                state = state_unstr(str_state)
                tmp = sum([decay(delta + 1) for delta in indexes(state, ac)])  # Recency function of (s,ac)
                x = np.array([tmp ** j for j in range(self.d + 1)]).reshape(self.d + 1)
                phi_x = phi(x, user, self.nUsers)
                phi_tilde_x = self.A_kro_inv_sqrt.dot(phi_x)
                # Expected reward
                _rewards[str_state][ac] = self.theta[ac].dot(phi_tilde_x)[0][0]
                # Confidence bound
                # _rewards[str_state][ac] += (self.conf[ac]*(phi_tilde_x.T@V_inv[ac]@phi_tilde_x))[0][0]
                # import ipdb;ipdb.set_trace()
                _rewards[str_state][ac] += (self.conf[ac] * np.sqrt((phi_tilde_x.T @ np.linalg.solve(self.V[ac], phi_tilde_x))))[0][0]

        # Apply Value Iteration
        for it in range(self.n_iter):
            shuffle(self.states)
            for str_state in self.states:
                Vf[str_state] = max(list(map(lambda a: self.Bellman_a(a, Vf, self.gamma, str_state, _rewards), range(self.nA))))

        # Find policy
        for str_state in self.states:
            state = state_unstr(str_state)
            r_arms = np.zeros(self.nA)
            for ac in range(self.nA):
                tmp = sum([decay(delta + 1) for delta in indexes(state, ac)])  # Compute decay
                x = np.array([tmp ** j for j in range(self.d + 1)]).reshape(1, self.d + 1)
                phi_x = phi(x[0], user, self.nUsers)
                phi_tilde_x = self.A_kro_inv_sqrt.dot(phi_x)
                r = self.theta[ac].dot(phi_tilde_x)[0][0] + (self.conf[ac] * np.sqrt((phi_tilde_x.T @ np.linalg.solve(self.V[ac], phi_tilde_x))))[0][0]
                next_state = self.transition(str_state, ac)
                r += self.gamma * Vf[next_state]
                r_arms[ac] = r
                # import ipdb;ipdb.set_trace()
            self.policy[user][str_state] = np.argmax(r_arms)
    
    def run(self,T=1000,lbda=0.5,gamma=1,n_iter=100):
        """
        Parameters
        ========
        T: int
            number of iterations of GoLinUCRL
        lbda: float
            regularization for linear model learning
        gamma: float
            discount parameter of Bellman operator
        n_iter:
            number of iterations of Bellman operator
        """
        
        self.V = dict.fromkeys(self.actions)
        self.X = dict.fromkeys(self.actions)
        self.b = dict.fromkeys(self.actions)
        for a in self.actions:
            self.V[a] = lbda*np.eye((self.d+1)*self.nUsers)
            self.X[a] = []
            self.b[a] = np.zeros(((self.d+1)*self.nUsers,1))

        self.gamma = gamma
        self.lbda=lbda
        self.update_time = []
        self.n_iter = n_iter
        
        while self.t.sum()<T:
            # Select uniformly randomly one user
            u=np.random.randint(self.nUsers)
            self.user_sequence.append(u)
            s=self.cur_states[u]
            
            # Choose action
            if self.t[u]<self.nA:
                a = self.t[u] # Pull each arm at least once FOR EACH USER
            else:
                a = self.policy[u][s]
            self.arm_sequence[u].append(a)
            
            self.roundN[a] += 1
            next_state = self.transition(s,a)
            
            # Observe reward
            r = self.reward(s,a,u)
            self.R[a].append(r)
            self.reward_sequence.append(r)
            
            # Update V, X, b, user state, t 
            state = state_unstr(s)
            tmp = sum([decay(delta+1) for delta in indexes(state,a)]) # Recency function of (s,a)
            x = np.array([tmp**j for j in range(self.d+1)]).reshape(self.d+1)
            phi_x = phi(x,u,self.nUsers)
            phi_tilde_x = self.A_kro_inv_sqrt.dot(phi_x)
            self.V[a] += phi_tilde_x.dot(phi_tilde_x.T)
            self.X[a].append(phi_tilde_x)
            self.b[a] += r*phi_tilde_x
            self.cur_states[u] = next_state
            self.t[u] += 1
            self.roundN[a] += 1
            
            # Update policy if we spent enough time in one action in the current round
            
            update_policy = False
            for ac in self.actions:
                if (self.roundN[ac] >= self.N[ac]) & (self.roundN[ac]>1):
                    update_policy = True
            
            if update_policy:
                for ac in self.actions:
                    self.N[ac] += self.roundN[ac]
                    self.roundN[ac] = 0
                self.round+=1
                self.update_time.append([round(e) for e in self.t.tolist()])
                
                # Compute new theta for each action
                #V_inv = dict.fromkeys(range(self.nA))
                for ac in self.actions:
                    #V_inv[ac] = np.linalg.inv(self.V[ac])
                    #self.theta[ac] = V_inv[ac].dot(self.b[ac])
                    #import ipdb;ipdb.set_trace()
                    self.theta[ac] = np.linalg.solve(self.V[ac],self.b[ac]).reshape(1,(self.d+1)*self.nUsers)
                    self.conf[ac] = self.conf_r(self.t.sum()+1,ac,lbda) # Not sure : t.sum() or t[u] ?
                    
                # Update the policy of each user
                for user in range(self.nUsers):
                    self.policyUpdate(user)
                    # print(user)
                    # print(self.policy[user])

                print('update policies, round=',self.round)
            else:
                print("not update")
            print("iteration number:",self.t)
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 14:29:52 2019

@author: Antoine
"""
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import math
from itertools import product as itp
from scipy.sparse.csgraph import laplacian
from GoLinUCRL import GLUCRL
from UCRL import UCRL, linUCRL
from copy import deepcopy
from sklearn.cluster import KMeans
import scipy.spatial.distance as sd
import ast
from random import shuffle
import datetime

#==============================================================================
## Cluster Users into nUsers Profiles
#==============================================================================

# Read datasets
nUsers = 4
path = "../ml-20m/"
ratings = pd.read_csv(path+'ratings.csv')
movies = pd.read_csv(path+'movies.csv')

# Restrict to users that watched at least 1000 movies
ratings = ratings[ratings['userId'].isin(ratings['userId'].value_counts()[ratings['userId'].value_counts()>2000].index)]

# Get the genre ratings
def get_genre_ratings(ratings, movies, genres, column_names):
    genre_ratings = pd.DataFrame()
    for genre in genres:        
        genre_movies = movies[movies['genres'].str.contains(genre)] # Get the movies of a given genre
        # Get average rating of this genre on all the different movies watched by each user, with 2 decimals
        avg_genre_votes_per_user = ratings[ratings['movieId'].isin(genre_movies['movieId'])].loc[:, ['userId', 'rating']].groupby(['userId'])['rating'].mean().round(2)
        # Concatenate these ratings for different genres iteratively in genre_ratings
        genre_ratings = pd.concat([genre_ratings, avg_genre_votes_per_user], axis=1)
        
    genre_ratings.columns = column_names
    return genre_ratings

Genres = ['Action','Adventure','Animation','Children','Comedy','Crime','Documentary','Drama','Fantasy','Horror','Musical','Mystery','Romance','Sci-Fi','Thriller','War','Western']
Genres = ['Action','Thriller', 'Romance', 'Comedy', 'Drama', 'Crime', 'Adventure'] # Only consider most represented genres for now
# Genres = ['Action','Thriller', 'Romance', 'Comedy'] # Only consider most represented genres for now

genre_ratings = get_genre_ratings(ratings, movies, Genres, ['avg_'+ genre +'_rating' for genre in Genres])
genre_ratings = genre_ratings.fillna(genre_ratings.mean())

# Create an instance of KMeans to find 10 clusters
kmeans = KMeans(n_clusters=nUsers)
assignments = kmeans.fit_predict(genre_ratings.values)
profiles = kmeans.cluster_centers_

#==============================================================================
## Least-Squares estimation of the oracle theta
#==============================================================================

ratings = ratings.merge(movies,on='movieId')
ratings = ratings.drop(columns=['title'])
w = 5
d = 5

def compute_theta(ratings, assignments, Genres, w, d):
    theta=np.zeros((nUsers,len(Genres),d+1))

    for cluster in range(nUsers): # Iterate on the different clusters
        # Get users assigned to cluster c
        users_cluster = genre_ratings.index[np.where(assignments == cluster)[0]]

        for genre_label, genre in enumerate(Genres): # Iterate on the genres
            # Collect equations for this genre on each user of this cluster
            rewards_cluster_genre = []
            recencies_cluster_genre = []

            for user in users_cluster:
                ratings_user = ratings[ratings['userId'] == user].sort_values(by = 'timestamp') # Ratings of this user
                rewards_user_genre = []
                recencies_user_genre = []
                actions_user = [] # Its last w values is the state, can be multiple genres

                for i in range(len(ratings_user)):  # Iterate on all the movies watched by user
                    current_movie = ratings_user.iloc[i] # Current movie
                    current_genres = current_movie.genres.split('|')  # Genres of the movie
                    if genre in current_genres: # If the Genre is the one we are collecting equations for ...
                        rewards_user_genre.append(current_movie.rating)  # Update rewards of the user for genre
                        current_recency = 0 # Initialize recency for the current movie
                        for j in range(1,min(i,w)+1): # Look at the last w (or i if there is not enough history) movies watched
                            if genre in set(actions_user[-j]):# If j time steps before, user has watched a movie of the genre genre
                                current_recency += 1 / j  # ... Then the recency is updated
                        recencies_user_genre.append(current_recency)  # Update recencies of the user for genre
                    actions_user.append(current_genres)  # Update actions

                rewards_cluster_genre.extend(rewards_user_genre) # Update rewards associated to the cluster
                recencies_cluster_genre.extend(recencies_user_genre) # Update recencies associated to the cluster

            # Least-Squares estimate of theta
            contexts_cluster_genre = [[recency ** j for j in range(d + 1)] for recency in recencies_cluster_genre]
            theta[cluster][genre_label] = np.linalg.lstsq(contexts_cluster_genre, rewards_cluster_genre, rcond = None)[0]
            print("Theta computed for user " + str(cluster) + " for genre " + genre)

    return(theta)

theta = compute_theta(ratings, assignments, Genres, w, d)
# np.save('thetaoracle.npy', theta)
# theta = np.load('thetaoracle.npy')

#==============================================================================
## Construction of the graph
#==============================================================================

nActions = len(Genres)
window = w
nStates = nActions**window
print('Space state size:',nStates) 
states = list(itp(range(nActions), repeat=window)) # all possibles states
states = list(map(lambda s: str(list(s)),states))

# Construct a k-NN graph 
def similarity_function(x,y,var):
    return np.exp(-sd.euclidean(x,y)**2/(2*var))

var = 1
W = np.zeros((nUsers,nUsers))
similarities = np.zeros((nUsers,nUsers))
for i in range(nUsers):
    for j in range(nUsers):
        similarities[i][j] = similarity_function(profiles[i], profiles[j], var)
        
k = 3
for i in range(nUsers):
    k_closest = np.argpartition(similarities[i], -k-1)[-k-1:]
    for j in k_closest:
        # Take the k closest excluding self-similarity
        if j!=i:
            W[i][j] = similarities[i][j]

D = np.diag(W.sum(axis=1))
L = D-W

Lid = np.zeros(nUsers) # TEST

alpha    = 2
lbda     = 0.5
gamma    = 0.99
T        = 200
n_iter   = 100
n_repeat = 1

# Transition probabilities
def next_state_bis(x,a):
    # x[j] contains the arm that have been played j+1 timesteps ago
    y = eval(deepcopy(x)) #FIX
    y = y[:-1]
    y = [int(a)] + y
    return str(y) #FIX

def f(delta):
    if delta==0:
        out = 0
    else:
        out = 1/delta
    return out

def indexes(l,elt):
    indices = [i for i, x in enumerate(l) if x == elt]
    return indices

def reward(x,a,user): 
    decay = np.sum([f(delta+1) for delta in indexes(x,a)]) # compute decay
    decay = np.array([decay**j for j in range(d+1)])
    return np.array(theta[user][int(a)]).dot(decay)+np.random.normal(0,0.1) #sample a reward #FIX INTA

def reward_nonoise(x,a,user):
    decay = np.sum([f(delta + 1) for delta in indexes(x, a)])  # compute decay
    decay = np.array([decay ** j for j in range(d + 1)])
    return np.array(theta[user][int(a)]).dot(decay)

def reward_one(user):
    return lambda x,a :reward(x,a,user)

run_all = True
#==============================================================================
# GoLinUCRL with graph L
#==============================================================================
print('\n Start GoLinUCRL with graph L')

mod_GLUCRL = dict.fromkeys(range(n_repeat))
for i in range(n_repeat):
    mod_GLUCRL[i] = GLUCRL(alpha,d,states,actions=range(nActions),reward=reward,transition=next_state_bis,L=L)
    mod_GLUCRL[i].run(T=T*nUsers,lbda=lbda,gamma=gamma,n_iter=n_iter)

#==============================================================================
# GoLinUCRL with empty graph
#==============================================================================
if run_all:
    print('\n Start GoLinUCRL with empty graph')

    mod_GLUCRL_empty = dict.fromkeys(range(n_repeat))
    for i in range(n_repeat):
        mod_GLUCRL_empty[i] = GLUCRL(alpha,d,states,actions=range(nActions),reward=reward,transition=next_state_bis,L=Lid)
        mod_GLUCRL_empty[i].run(T=T*nUsers,lbda=lbda,gamma=gamma,n_iter=n_iter)

#==============================================================================
# LinUCRL
#==============================================================================
if run_all:
    print('\n Start linUCRL')

    mod_linucrl = dict.fromkeys(range(n_repeat))
    for i in range(n_repeat):
        mod_linucrl[i] = dict.fromkeys(range(nUsers))
        for u in range(nUsers):
            mod_linucrl[i][u] = linUCRL(alpha=alpha,d=d,states=states,s0=states[0],actions=range(nActions),
                            reward=reward_one(u),transition=next_state_bis)
            mod_linucrl[i][u].run(T=T,gamma=gamma,n_iter=n_iter)

#==============================================================================
# Optimal policy - Oracle
#==============================================================================
if run_all:
    # Transition probabilities
    def next_state_(x,a):
        # x[j] contains the arm that have been played j+1 timesteps ago
        y = deepcopy(x)
        y = y[:-1]
        y = [a] + y
        return y

    # Bellman Operator
    def Bellman_a(a,gamma,state,user):
        state = ast.literal_eval(state)
        # update for one state at random
        r = reward(state,a,user) # we play a in this state
        next_state = next_state_(state,a)
        r+=gamma*V[user][str(next_state)]
        return r

    # Initialize Value Function
    V = dict.fromkeys(range(nUsers))
    S = list(itp(range(len(Genres)), repeat=window)) # all possibles states
    S = list(map(lambda s: str(list(s)),S))
    for user in range(nUsers):
        V[user] = dict.fromkeys(list(map(str, states)))
        for state in S:
            V[user][state] = 0

    # Value Iteration for n_iter steps
    for it in range(n_iter):
        for user in range(nUsers):
            shuffle(S)
            for state in S:
                V[user][state] = max(list(map(lambda a: Bellman_a(a,gamma,state,user),range(len(Genres)))))

    # Compute Optimal Policy
    policy = dict.fromkeys(range(nUsers))
    for user in range(nUsers):
        policy[user] = dict.fromkeys(list(map(str, states)))
        for state in V[user].keys():
            state = ast.literal_eval(state)
            tmp = np.zeros(len(Genres))
            for a in range(len(Genres)):
                r = reward_nonoise(state,a,user) # Reward without Noise
                next_state = next_state_(state,a)
                r+=gamma*V[user][str(next_state)]
                tmp[a] = r
            policy[user][str(state)] = np.argmax(tmp)

    # Apply Optimal Policy
    optimal_rewards = dict.fromkeys(range(nUsers))
    optimal_arms = dict.fromkeys(range(nUsers))
    for user in range(nUsers):
        optimal_rewards[user] = dict.fromkeys(range(n_repeat))
        optimal_arms[user] = [] # Same for all of the n_repeat tests

        for i in range(n_repeat):
            optimal_rewards[user][i] = []
            state = ast.literal_eval(S[0]) # Start from state 0

            for t in range(T):
                a = policy[user][str(state)]
                r = reward(state,a,user) # Noisy Reward
                state = next_state_(state,a)
                optimal_rewards[user][i].append(r)
                optimal_arms[user].append(a)

#==============================================================================
# Cumulated Mean Reward
#==============================================================================
time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
save = "exp_T%i_k%i_nrepeat%i_nclusters%i_nactions%i_var%f_%s/" % (T,k,n_repeat,nUsers,nActions, var, time)
os.makedirs(save)

colors = ['blue','green','hotpink','darkcyan','goldenrod','grey','brown','black','purple','yellow','orange']

golinucrl_cumreward = np.cumsum(mod_GLUCRL[0].reward_sequence) / (n_repeat*(np.arange(1,1+len(mod_GLUCRL[0].reward_sequence))))
for i in range(1, n_repeat):
    golinucrl_cumreward += np.cumsum(mod_GLUCRL[i].reward_sequence) / (n_repeat*(np.arange(1,1+len(mod_GLUCRL[i].reward_sequence))))
np.save(save+ "golinucrl_cumreward.npy", golinucrl_cumreward)
np.save(save + "golinucrl_update_time.npy",mod_GLUCRL[0].update_time)

if run_all:
    golinucrl_empty_cumreward = np.cumsum(mod_GLUCRL_empty[0].reward_sequence) / (n_repeat*(np.arange(1,1+len(mod_GLUCRL_empty[0].reward_sequence))))
    for i in range(1, n_repeat):
        golinucrl_empty_cumreward += np.cumsum(mod_GLUCRL_empty[i].reward_sequence) / (n_repeat*(np.arange(1,1+len(mod_GLUCRL_empty[i].reward_sequence))))
    np.save(save+"golinucrl_empty_cumreward.npy", golinucrl_empty_cumreward)
    golinucrl_empty_update_time = mod_GLUCRL_empty[0].update_time
    np.save(save + "golinucrl_empty_update_time.npy", golinucrl_empty_update_time)
    golinucrl_empty_user_sequence = mod_GLUCRL_empty[0].user_sequence
    np.save(save + "golinucrl_empty_user_sequence.npy", golinucrl_empty_user_sequence)
    golinucrl_empty_reward_sequence = mod_GLUCRL_empty[0].reward_sequence
    np.save(save + "golinucrl_empty_reward_sequence.npy", golinucrl_empty_reward_sequence)
    golinucrl_empty_arm_sequence = dict.fromkeys(range(nUsers))

    linucrl_cumreward = dict.fromkeys(range(nUsers))
    linucrl_update_time = dict.fromkeys(range(nUsers))
    linucrl_arm_sequence = dict.fromkeys(range(nUsers))
    for c in range(nUsers):
        golinucrl_empty_arm_sequence[c] = mod_GLUCRL_empty[0].arm_sequence[c]
        np.save(save + "golinucrl_empty_arm_sequence_%i.npy" % c, golinucrl_empty_arm_sequence[c])
        linucrl_cumreward[c] = np.cumsum(mod_linucrl[0][c].reward_sequence) / (n_repeat * (np.arange(1, 1 + len(mod_linucrl[0][c].reward_sequence))))
        for i in range(1, n_repeat):
            linucrl_cumreward[c] += np.cumsum(mod_linucrl[i][c].reward_sequence) / (n_repeat * (np.arange(1, 1 + len(mod_linucrl[i][c].reward_sequence))))
        np.save(save + "linucrl_cumreward"+str(c)+".npy", linucrl_cumreward[c])
        linucrl_update_time[c] = mod_linucrl[0][c].update_time
        np.save(save + "linucrl%i_update_time.npy" %c,linucrl_update_time[c])
        linucrl_arm_sequence[c] = mod_linucrl[0][c].arm_sequence
        np.save(save + "linucrl%i_arm_sequence.npy" % c, linucrl_arm_sequence[c])

    oracle_cumreward = dict.fromkeys(range(nUsers))
    for c in range(nUsers):
        oracle_cumreward[c] = np.cumsum(optimal_rewards[c][0]) / (n_repeat * (np.arange(1, 1 + len(optimal_rewards[c][0]))))
        for i in range(1, n_repeat):
            oracle_cumreward[c] += np.cumsum(optimal_rewards[c][i]) / (n_repeat * (np.arange(1, 1 + len(optimal_rewards[c][0]))))
        np.save(save + "oracle_cumreward" + str(c) + ".npy", oracle_cumreward[c])
        np.save(save + "optimal_arms"+ str(c) + ".npy", optimal_arms[c])

else:
    dir = "exp_T200_k2_nrepeat1_nclusters4_nactions4_var0.500000_20191216_105657/"
    golinucrl_empty_cumreward = np.load(dir + "golinucrl_cumreward.npy")
    golinucrl_empty_update_time = np.load(dir + "golinucrl_empty_update_time.npy")
    golinucrl_empty_user_sequence = np.load(dir + "golinucrl_empty_user_sequence.npy")
    golinucrl_empty_reward_sequence = np.load(dir + "golinucrl_empty_reward_sequence.npy")
    golinucrl_empty_arm_sequence = dict.fromkeys(range(nUsers))
    linucrl_cumreward = dict.fromkeys(range(nUsers))
    oracle_cumreward = dict.fromkeys(range(nUsers))
    optimal_arms = dict.fromkeys(range(nUsers))
    linucrl_update_time = dict.fromkeys(range(nUsers))
    linucrl_arm_sequence = dict.fromkeys(range(nUsers))
    for c in range(nUsers):
        golinucrl_empty_arm_sequence[c] = np.load(dir + "golinucrl_empty_arm_sequence_%i.npy" %c)
        linucrl_cumreward[c] = np.load(dir + "linucrl_cumreward%i.npy" %c)
        linucrl_update_time[c] = np.load(dir + "linucrl%i_update_time.npy" %c)
        oracle_cumreward[c] = np.load(dir + "oracle_cumreward%i.npy" %c)
        optimal_arms[c] = np.load(dir + "optimal_arms"+ str(c) + ".npy")
        linucrl_arm_sequence[c] = np.load(dir + "linucrl%i_arm_sequence.npy" %c)

#==============================================================================
# Cumulated Mean Reward Plots
#==============================================================================

plt.plot(golinucrl_empty_cumreward, color = colors[0], label = "GoLinUCRL with empty graph") #[:nUsers*T]
plt.scatter(np.sum(golinucrl_empty_update_time,axis = 1), golinucrl_empty_cumreward[np.sum(golinucrl_empty_update_time,axis = 1)], color = colors[0])
plt.plot(golinucrl_cumreward, color = colors[1], label = "GoLinUCRL")
plt.scatter(np.sum(mod_GLUCRL[0].update_time,axis = 1), golinucrl_cumreward[np.sum(mod_GLUCRL[0].update_time,axis = 1)], color = colors[1])
for c in range(nUsers):
    plt.plot(linucrl_cumreward[c], color = colors[2], label = "LinUCRL for user "+str(c))
    plt.scatter(linucrl_update_time[c], linucrl_cumreward[c][linucrl_update_time[c]], color = colors[2])
    plt.plot(oracle_cumreward[c], label = "Oracle for user "+str(c), color = colors[3])
plt.xlabel("Time Steps")
plt.ylabel("Mean Cumulated Reward")
plt.legend(loc = "best")
plt.savefig(save+'performance.pdf')
plt.show()

#==============================================================================
# Cumulated Mean Reward for each user Plots
#==============================================================================

for c in range(nUsers):
    index_cluster_empty = np.where(np.array(golinucrl_empty_user_sequence) == c)[0]
    rewards_cluster_empty = np.array(golinucrl_empty_reward_sequence)[index_cluster_empty]
    index_cluster = np.where(np.array(mod_GLUCRL[0].user_sequence) == c)[0]
    rewards_cluster = np.array(mod_GLUCRL[0].reward_sequence)[index_cluster]
    cumrewards_cluster_empty = np.cumsum(rewards_cluster_empty) / (np.arange(1,1+len(rewards_cluster_empty)))
    cumrewards_cluster = np.cumsum(rewards_cluster) / (np.arange(1,1+len(rewards_cluster)))
    plt.plot(cumrewards_cluster_empty, label = "GoLinUCRL with empty graph", color = colors[0])
    plt.scatter(np.array(golinucrl_empty_update_time)[:,c],
                cumrewards_cluster_empty[np.array(golinucrl_empty_update_time)[:,c]], color=colors[0])
    plt.plot(cumrewards_cluster, label = "GoLinUCRL with graph L", color = colors[1])
    plt.scatter(np.array(mod_GLUCRL[0].update_time)[:, c],
                cumrewards_cluster[np.array(mod_GLUCRL[0].update_time)[:, c]], color=colors[1])
    plt.plot(linucrl_cumreward[c], label = "LinUCRL", color = colors[2])
    plt.scatter(linucrl_update_time[c], linucrl_cumreward[c][linucrl_update_time[c]], color=colors[2])
    plt.plot(oracle_cumreward[c], label = "Oracle", color = colors[3])
    plt.legend(loc = "best")
    plt.savefig(save + 'performance_cluster%i.pdf' %c)
    plt.show()

#==============================================================================
# Strategies Plots
#==============================================================================

for c in range(nUsers):
    fig = plt.figure(figsize=(8, 8))
    sub1 = plt.subplot(5, 1, 1)
    sub1.plot(range(40),[x for x in optimal_arms[c][-40:]],'o-',color=colors[0],linewidth=3,
              markersize=10,label='optimal oracle')
    plt.yticks(range(nActions),Genres)
    lgd = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    sub2 = plt.subplot(5, 1, 2)
    sub2.plot(range(40),[x for x in linucrl_arm_sequence[c][-40:]],'o-',color=colors[1],linewidth=3,
              markersize=10,label='linUCRL')
    plt.yticks(range(nActions),Genres)
    lgd = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    sub3 = plt.subplot(5, 1, 3)
    sub3.plot(range(40),[x for x in golinucrl_empty_arm_sequence[c][-40:]],'o-',color=colors[2],linewidth=3,
              markersize=10,label='GoLinUCRL with empty graph')
    plt.yticks(range(nActions),Genres)
    lgd = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    sub4 = plt.subplot(5, 1, 4)
    sub4.plot(range(40),[x for x in mod_GLUCRL[0].arm_sequence[c][-40:]],'o-',color=colors[3],linewidth=3,
              markersize=10,label='GoLinUCRL with graph L')
    plt.yticks(range(nActions),Genres)
    lgd = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    # fig.tight_layout()
    plt.savefig(save + 'strategies_cluster%i.pdf' % c,bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.show()

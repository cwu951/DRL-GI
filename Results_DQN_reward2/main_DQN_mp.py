# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 20:39:01 2022

@author: chong
"""

import sys 
sys.path.append("..") 

import numpy as np
import pandas as pd
from SWMM import SWMM_ENV
from DQN import Buffer
from DQN import DQN as DQN
import tensorflow as tf
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import os


Train=True
init_train=True

tf.compat.v1.reset_default_graph()

env_params={
        'orf':os.path.dirname(os.getcwd())+'/SWMM/chaohu',
        'orf_save':'chaohu_RTC',
        'parm':os.path.dirname(os.getcwd())+'/states_yaml/chaohu',
        'advance_seconds':300,
        'kf':1,
        'kc':1,
        'reward_type':'2'
    }
env=SWMM_ENV.SWMM_ENV(env_params)

raindata = np.load(os.path.dirname(os.getcwd())+'/rainfall/training_raindata.npy').tolist()


agent_params={
    'state_dim':len(env.config['states']),
    'action_dim':2**len(env.config['action_assets']),

    'encoding_layer':[50,50,50],
    'value_layer':[50,50,50],
    'advantage_layer':[50,50,50],
    'num_rain':50,

    'train_iterations':20,
    'training_step':500,
    'gamma':0.01,
    'epsilon':0.5,
    'ep_min':0,
    'ep_decay':0.9,
    'learning_rate':0.0001,

    'action_table':pd.read_csv(os.path.dirname(os.getcwd())+'/SWMM/action_table.csv').values[:,1:],
}

model = DQN.DQN(agent_params)
if init_train:
    model.model.save_weights('./model/dqn.h5')    
    model.target_model.save_weights('./model/target_dqn.h5')    
model.load_model('./model/')
print('model done')

###############################################################################
# Train
###############################################################################
    
def interact(i,ep):
    env=SWMM_ENV.SWMM_ENV(env_params)
    tem_model = DQN.DQN(agent_params)
    tem_model.load_model('./model/')
    tem_model.params['epsilon']=ep
    s,a,r,s_ = [],[],[],[]
    observation, episode_return, episode_length = env.reset(raindata[i],i,True,os.path.dirname(os.getcwd())), 0, 0
    
    done = False
    while not done:
        # Get the action, and take one step in the environment
        observation = np.array(observation).reshape(1, -1)
        action = DQN.sample_action(observation,tem_model,True)
        at = tem_model.action_table[int(action[0].numpy())].tolist()
        observation_new, reward, results, done = env.step(at)
        episode_return += reward
        episode_length += 1

        # Store obs, act, rew
        # buffer.store(observation, action, reward, value_t, logprobability_t)
        s.append(observation)
        a.append(action)
        r.append(reward)
        s_.append(observation_new)
        
        # Update the observation
        observation = observation_new
    # Finish trajectory if reached to a terminal state
    last_value = 0 if done else tem_model.predict(observation.reshape(1, -1))
    return s,a,r,s_,last_value,episode_return,episode_length

if Train:
    #tf.config.experimental_run_functions_eagerly(True)

    # main training process   
    history = {'episode': [], 'Batch_reward': [], 'Episode_reward': [], 'Loss': []}
    
    # Iterate over the number of epochs
    for epoch in range(model.params['training_step']):
        # Initialize the sum of the returns, lengths and number of episodes for each epoch
        sum_return = 0
        sum_length = 0
        num_episodes = 0
        
        # Initialize the buffer
        buffer = Buffer.Buffer(model.params['state_dim'], int(len(raindata[0])*model.params['num_rain']))
        
        # Iterate over the steps of each epoch
        # Parallel method in joblib
        res = Parallel(n_jobs=20)(delayed(interact)(i,model.params['epsilon']) for i in range(model.params['num_rain'])) 
        
        for i in range(model.params['num_rain']):
            #s, a, r, vt, lo, lastvalue in buffer
            for o,a,r,o_ in zip(res[i][0],res[i][1],res[i][2],res[i][3]):
                buffer.store(o,a,r,o_)
            buffer.finish_trajectory(res[i][4])
            sum_return += res[i][5]
            sum_length += res[i][6]
            num_episodes += 1
        
        # Get values from the buffer
        (
            observation_buffer,
            action_buffer,
            observation_next_buffer,
            reward_buffer,
            advantage_buffer,
        ) = buffer.get()

        # Update the policy and implement early stopping using KL divergence
        for _ in range(model.params['train_iterations']):
            DQN.train_value(observation_buffer, action_buffer, reward_buffer, observation_next_buffer, model)
            
        model.model.save_weights('./model/dqn.h5')
        model.model.save_weights('./model/target_dqn.h5')    
        # log training results
        history['episode'].append(epoch)
        history['Episode_reward'].append(sum_return)
        # reduce the epsilon egreedy and save training log
        if model.params['epsilon'] >= model.params['ep_min'] and epoch % 3 == 0:
            model.params['epsilon'] *= model.params['ep_decay']
            #model.params['ep_decay'] *= model.params['ep_min']
        
        # Print mean return and length for each epoch
        print(
            f" Epoch: {epoch + 1}. Return: {sum_return}. Mean Length: {sum_length / num_episodes}"
        )
        
        np.save('./Results/Train.npy',history)
    
    # plot
    plt.figure()
    plt.plot(history['Episode_reward'])
    plt.savefig('./Results//Train.tif')

   
###############################################################################
# end Train
###############################################################################

# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 20:39:01 2022

@author: chong
"""

import sys
sys.path.append("..")

import numpy as np
import pandas as pd
from SWMM_GR import SWMM_ENV_GI_utilization as SWMM_ENV
from PPO import Buffer
from PPO import PPO as PPO
import tensorflow as tf
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


Train = True
init_train = True

tf.compat.v1.reset_default_graph()

BASE_DIR = os.path.dirname(os.getcwd())
RESULT_DIR = "./Results"
MODEL_DIR = "./model"

os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

env_params = {
    'orf': BASE_DIR + '/SWMM_GR/chaohu',
    'orf_save': 'chaohu_GI_RTC',
    'parm': BASE_DIR + '/states_yaml/chaohu',
    'advance_seconds': 300,
    'kf': 1,
    'kc': 1,
    'reward_type': '3',
    'run_baseline': True,
    'run_gi_only': True
}
env = SWMM_ENV.SWMM_ENV(env_params)

raindata = np.load(BASE_DIR + '/rainfall/training_raindata.npy').tolist()

agent_params = {
    'state_dim': len(env.config['states']),
    'action_dim': int(2 ** len(env.config['action_assets'])),
    'actornet_layer': [30, 30, 30],
    'criticnet_layer': [30, 30, 30],

    'bound_low': 0,
    'bound_high': 1,

    'clip_ratio': 0.01,
    'target_kl': 0.03,
    'lam': 0.01,

    'policy_learning_rate': 0.005,
    'value_learning_rate': 0.005,
    'train_policy_iterations': 50,
    'train_value_iterations': 50,

    'num_rain': 50,
    'training_step': 2000,
    'gamma': 0.01,
    'epsilon': 1,
    'ep_min': 1e-50,
    'ep_decay': 0.5,

    'action_table': pd.read_csv(BASE_DIR + '/SWMM_GR/action_table.csv').values[:, 1:],
}

model = PPO.PPO(agent_params)
if init_train:
    model.critic.save_weights(MODEL_DIR + '/PPOcritic.h5')
    model.actor.save_weights(MODEL_DIR + '/PPOactor.h5')
model.load_model(MODEL_DIR + '/')
print('model done')


###############################################################################
# Train
###############################################################################

def interact(i, ep):
    local_env_params = env_params.copy()
    local_env_params['process_id'] = i

    env = SWMM_ENV.SWMM_ENV(local_env_params)
    tem_model = PPO.PPO(agent_params)
    tem_model.load_model(MODEL_DIR + '/')
    tem_model.params['epsilon'] = ep

    s, a, r, vt, lo = [], [], [], [], []
    observation, episode_return, episode_length = env.reset(
        raindata[i], i, True, BASE_DIR
    ), 0, 0

    done = False
    while not done:
        observation = np.array(observation).reshape(1, -1)
        logits, action = PPO.sample_action(observation, tem_model, True)
        at = tem_model.action_table[int(action[0].numpy())].tolist()
        observation_new, reward, results, done = env.step(at)
        episode_return += reward
        episode_length += 1

        value_t = tem_model.critic(observation)
        logprobability_t = PPO.logprobabilities(
            logits, action, tem_model.params['action_dim']
        )

        s.append(observation)
        a.append(action)
        r.append(reward)
        vt.append(value_t)
        lo.append(logprobability_t)

        observation = observation_new

    last_value = 0 if done else tem_model.critic(observation.reshape(1, -1))

    episode_reward3 = sum(env.results.get('reward3_list', []))
    episode_r1 = sum(env.results.get('r1_list', []))

    return (
        s,
        a,
        r,
        vt,
        lo,
        last_value,
        episode_return,
        episode_length,
        episode_reward3,
        episode_r1,
    )


if Train:
    tf.config.experimental_run_functions_eagerly(True)

    history = {
        'episode': [],
        'Batch_reward': [],
        'Episode_reward': [],
        'Episode_reward3': [],
        'Episode_r1': [],
        'Loss': []
    }

    for epoch in range(model.params['training_step']):
        sum_return = 0
        sum_length = 0
        num_episodes = 0

        sum_reward3 = 0
        sum_r1 = 0

        buffer = Buffer.Buffer(
            model.params['state_dim'],
            int(len(raindata[0]) * model.params['num_rain'])
        )

        res = Parallel(n_jobs=30)(
            delayed(interact)(i, model.params['epsilon'])
            for i in range(model.params['num_rain'])
        )

        for i in range(model.params['num_rain']):
            for o, a, r, vt, lo in zip(
                res[i][0], res[i][1], res[i][2], res[i][3], res[i][4]
            ):
                buffer.store(o, a, r, vt, lo)

            buffer.finish_trajectory(res[i][5])
            sum_return += res[i][6]
            sum_length += res[i][7]
            num_episodes += 1

            sum_reward3 += res[i][8]
            sum_r1 += res[i][9]

        (
            observation_buffer,
            action_buffer,
            advantage_buffer,
            return_buffer,
            logprobability_buffer,
        ) = buffer.get()

        for _ in range(model.params['train_policy_iterations']):
            kl = PPO.train_policy(
                observation_buffer,
                action_buffer,
                logprobability_buffer,
                advantage_buffer,
                model
            )

        for _ in range(model.params['train_value_iterations']):
            PPO.train_value_function(observation_buffer, return_buffer, model)

        model.critic.save_weights(MODEL_DIR + '/PPOcritic.h5')
        model.actor.save_weights(MODEL_DIR + '/PPOactor.h5')

        if (epoch + 1) % 50 == 0:
            model.critic.save_weights(MODEL_DIR + '/PPOcritic_' + str(epoch + 1) + '.h5')
            model.actor.save_weights(MODEL_DIR + '/PPOactor_' + str(epoch + 1) + '.h5')

        history['episode'].append(epoch)
        history['Episode_reward'].append(sum_return)
        history['Episode_reward3'].append(sum_reward3)
        history['Episode_r1'].append(sum_r1)

        if model.params['epsilon'] >= model.params['ep_min'] and epoch % 10 == 0:
            model.params['epsilon'] *= model.params['ep_decay']

        print(
            f" Epoch: {epoch + 1}. Return: {sum_return}. "
            f"Reward3: {sum_reward3}. R1: {sum_r1}. "
            f"Mean Length: {sum_length / num_episodes}. Epsilon: {model.params['epsilon']}"
        )

        np.save(RESULT_DIR + '/Train_GI.npy', history)

    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    axes[0].plot(history['Episode_reward'], label='Combined Reward (0.7*r3 + 0.3*r1)')
    axes[0].set_ylabel('Cumulative Reward')
    axes[0].set_title('Combined Reward per Epoch')
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(history['Episode_reward3'], label='Reward3 (unweighted)', color='tab:orange')
    axes[1].set_ylabel('Cumulative Reward3')
    axes[1].set_title('Reward3 Component per Epoch')
    axes[1].legend()
    axes[1].grid(True)

    axes[2].plot(history['Episode_r1'], label='R1 (unweighted)', color='tab:green')
    axes[2].set_ylabel('Cumulative R1')
    axes[2].set_xlabel('Epoch')
    axes[2].set_title('R1 Component per Epoch')
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout()
    plt.savefig(RESULT_DIR + '/Train_GI.tif')

    np.save(RESULT_DIR + '/Train_GI_combined_reward.npy', history['Episode_reward'])
    np.save(RESULT_DIR + '/Train_GI_reward3.npy', history['Episode_reward3'])
    np.save(RESULT_DIR + '/Train_GI_r1.npy', history['Episode_r1'])


###############################################################################
# end Train
###############################################################################

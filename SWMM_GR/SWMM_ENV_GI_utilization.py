# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 22:41:39 2022

@author: chong

SWMM environment
can be used for any inp file
established based pyswmm
"""
import os
os.environ['CONDA_DLL_SEARCH_MODIFICATION_ENABLE']="1"
import numpy as np

from swmm_api.input_file import read_inp_file
from pyswmm import Simulation,Links,Nodes,RainGages,SystemStats
from swmm_api.input_file.sections.others import TimeseriesData
from swmm_api.input_file.section_labels import TIMESERIES

import matplotlib.pyplot as plt
import datetime
import yaml
import shutil 


def get_step_results(results,nodes,links,rgs,sys,config,params):
    
    # Calculate reward
    delt_flooding,delt_CSO,CSOtem,inflow=0,0,0,0
    for _temp in config['reward_targets']:
        if _temp[1] == 'flooding':
            if _temp[0] == 'system':
                delt_flooding += sys.routing_stats[_temp[1]] - results['flooding'][-1]
            else:
                delt_flooding += nodes[_temp[0]].statistics['flooding_volume']
            results['flooding'].append(sys.routing_stats[_temp[1]])
        else:
            CSOtem += nodes[_temp[0]].cumulative_inflow
    delt_CSO = CSOtem - results['CSO'][-1]
    results['CSO'].append(CSOtem)
            
    Qtw = sys.routing_stats['wet_weather_inflow']
    delt_Qtw = Qtw - results['inflow'][-1]
    results['inflow'].append(Qtw)
    
    #flooding time and cso time
    floodingt, delt_flooding_time = 0,0
    for n in nodes:
        if n.statistics['flooding_duration'] >= floodingt:
            floodingt = n.statistics['flooding_duration']
    delt_flooding_time = floodingt - results['total_flooding_time'][-1]
    results['total_flooding_time'].append(floodingt)

    delt_cso_time = 0
    if delt_CSO > 0:
        delt_cso_time = params['advance_seconds']/3600
    results['total_CSO_time'].append(results['total_CSO_time'][-1]+delt_cso_time)

    # Calculate original reward3
    if delt_Qtw == 0:
        reward3 = 0
    else:
        reward3 = 1 / (1 + delt_flooding/delt_Qtw + delt_CSO/delt_Qtw) - 1
    
    return results, reward3, delt_flooding, delt_CSO, delt_Qtw


class SWMM_ENV:
    def __init__(self,params): 
        self.params = params
        self.config = yaml.load(open(self.params['parm']+".yaml"), yaml.FullLoader)

        # Flag to determine if baseline should be executed
        self.run_baseline = params.get('run_baseline', False)
        self.run_gi_only = params.get('run_gi_only', False)
        
        #  Get process ID (for multi-process environments)
        self.process_id = params.get('process_id', 0)
        
        # Store simulators for baseline and GI-only cases
        self.baseline_sim = None
        self.gi_only_sim = None
        
        # Flag to indicate if simulation is finished
        self.baseline_done = False
        self.gi_only_done = False

        # Store cumulative values for integral calculation
        self.integral_data = None
    
    def reset(self,rain,rainid,trainlog,root):
        
        # Add unique identifier for multi-process environment
        unique_id = f"{self.process_id}_{rainid}"
        
        # Main simulation (with GI and DQN)
        if trainlog:
            root_gi = root + '/SWMM_GR/_teminp'
            root_base = root + '/SWMM/_teminp'
        else:
            root_gi = root + '/SWMM_GR/_temtestinp'
            root_base = root + '/SWMM/_temtestinp'
        
        
        os.makedirs(root_gi, exist_ok=True)
        os.makedirs(root_base, exist_ok=True)
            
        
        inp = read_inp_file(self.params['orf']+'.inp')
        inp[TIMESERIES]['rainfall']=TimeseriesData('rainfall',rain)
        main_inp_path = f"{root_gi}/{self.params['orf_save']}_{unique_id}_rain.inp"
        inp.write_file(main_inp_path)
        self.sim=Simulation(main_inp_path)
        self.sim.start()
        
        
        self.baseline_done = False
        self.gi_only_done = False
        
        # Setup baseline case (No GI, No DQN)
        if self.run_baseline:
            try:
                baseline_orf = self.params['orf'].replace('/SWMM_GR/', '/SWMM/')
                inp_base = read_inp_file(baseline_orf+'.inp')
                inp_base[TIMESERIES]['rainfall']=TimeseriesData('rainfall',rain)
                base_inp_path = f"{root_base}/baseline_{unique_id}_rain.inp"
                inp_base.write_file(base_inp_path)
                self.baseline_sim = Simulation(base_inp_path)
                self.baseline_sim.start()
            except Exception as e:
                print(f"Warning: Failed to initialize baseline simulation: {e}")
                self.baseline_sim = None
                self.run_baseline = False
        
        # Setup GI-only case (With GI, No DQN)
        if self.run_gi_only:
            try:
                inp_gi = read_inp_file(self.params['orf']+'.inp')
                inp_gi[TIMESERIES]['rainfall']=TimeseriesData('rainfall',rain)
                gi_inp_path = f"{root_gi}/gi_only_{unique_id}_rain.inp"
                inp_gi.write_file(gi_inp_path)
                self.gi_only_sim = Simulation(gi_inp_path)
                self.gi_only_sim.start()
            except Exception as e:
                print(f"Warning: Failed to initialize GI-only simulation: {e}")
                self.gi_only_sim = None
                self.run_gi_only = False

        if self.params['advance_seconds'] is None:
            self.sim._model.swmm_step()
            if self.baseline_sim:
                self.baseline_sim._model.swmm_step()
            if self.gi_only_sim:
                self.gi_only_sim._model.swmm_step()
        else:
            self.sim._model.swmm_stride(self.params['advance_seconds'])
            if self.baseline_sim:
                self.baseline_sim._model.swmm_stride(self.params['advance_seconds'])
            if self.gi_only_sim:
                self.gi_only_sim._model.swmm_stride(self.params['advance_seconds'])
                
        # Initial states
        nodes = Nodes(self.sim)
        links = Links(self.sim)
        rgs = RainGages(self.sim)
        states = []
        for _temp in self.config["states"]:
            if _temp[1] == 'depthN':
                states.append(nodes[_temp[0]].depth)
            elif _temp[1] == 'flow':
                states.append(links[_temp[0]].flow)
            elif _temp[1] == 'inflow':
                states.append(nodes[_temp[0]].total_inflow)
            else:
                states.append(rgs[_temp[0]].rainfall)

        # Initialize results
        self.results = {}
        self.results['CSO'], self.results['flooding'], self.results['inflow'] = [0], [0], [0]
        self.results['total_flooding_time'], self.results['total_CSO_time'] = [0], [0]
        self.results['res'] = [0]
        self.results['state'], self.results['action'], self.results['rewards'] = [], [], []

        # Reset integral data
        self.integral_data = {
            'base_flooding': [0],
            'base_cso': [0],
            'base_inflow': [0],
            'gi_only_flooding': [0],
            'gi_only_cso': [0],
            'gi_only_inflow': [0],
            'gi_dqn_flooding': [0],
            'gi_dqn_cso': [0],
            'gi_dqn_inflow': [0],
            'delta_1_integral': 0,
            'delta_2_integral': 0
        }
        
        return states
        
    def step(self,action):
        # Get main simulation results
        nodes = Nodes(self.sim)
        links = Links(self.sim)
        rgs = RainGages(self.sim)
        sys = SystemStats(self.sim)
        
        states = []
        for _temp in self.config["states"]:
            if _temp[1] == 'depthN':
                states.append(nodes[_temp[0]].depth)
            elif _temp[1] == 'flow':
                states.append(links[_temp[0]].flow)
            elif _temp[1] == 'inflow':
                states.append(nodes[_temp[0]].total_inflow)
            else:
                states.append(rgs[_temp[0]].rainfall)
            
        # Set DQN control actions (only for main simulation)
        for item,a in zip(self.config['action_assets'],action):
            links[item].target_setting = a
        
        # Advance all simulations
        done = False
        try:
            if self.params['advance_seconds'] is None:
                time = self.sim._model.swmm_step()
                if self.baseline_sim and not self.baseline_done:
                    time_base = self.baseline_sim._model.swmm_step()
                    if time_base <= 0:
                        self.baseline_done = True
                if self.gi_only_sim and not self.gi_only_done:
                    time_gi = self.gi_only_sim._model.swmm_step()
                    if time_gi <= 0:
                        self.gi_only_done = True
            else:
                time = self.sim._model.swmm_stride(self.params['advance_seconds'])
                if self.baseline_sim and not self.baseline_done:
                    time_base = self.baseline_sim._model.swmm_stride(self.params['advance_seconds'])
                    if time_base <= 0:
                        self.baseline_done = True
                if self.gi_only_sim and not self.gi_only_done:
                    time_gi = self.gi_only_sim._model.swmm_stride(self.params['advance_seconds'])
                    if time_gi <= 0:
                        self.gi_only_done = True
            
            done = False if time > 0 else True
        except Exception as e:
            print(f"Error in simulation step: {e}")
            done = True

        # Get results and reward3 from main simulation
        self.results, reward3, delt_flooding_dqn, delt_cso_dqn, delt_inflow_dqn = get_step_results(
            self.results, nodes, links, rgs, sys, self.config, self.params
        )

        # Calculate new reward r1
        reward = reward3  # Default: reward3
        
        if self.run_baseline and self.run_gi_only and self.baseline_sim and self.gi_only_sim:
            try:
                # Get baseline case results
                nodes_base = Nodes(self.baseline_sim)
                sys_base = SystemStats(self.baseline_sim)
                base_flooding = sys_base.routing_stats['flooding']
                base_cso = sum([nodes_base[_temp[0]].cumulative_inflow 
                               for _temp in self.config['reward_targets'] 
                               if _temp[1] != 'flooding'])
                base_inflow = sys_base.routing_stats['wet_weather_inflow']
                
                # Get GI-only case results
                nodes_gi = Nodes(self.gi_only_sim)
                sys_gi = SystemStats(self.gi_only_sim)
                gi_flooding = sys_gi.routing_stats['flooding']
                gi_cso = sum([nodes_gi[_temp[0]].cumulative_inflow 
                             for _temp in self.config['reward_targets'] 
                             if _temp[1] != 'flooding'])
                gi_inflow = sys_gi.routing_stats['wet_weather_inflow']
                
                # Record current values
                self.integral_data['base_flooding'].append(base_flooding)
                self.integral_data['base_cso'].append(base_cso)
                self.integral_data['base_inflow'].append(base_inflow)
                self.integral_data['gi_only_flooding'].append(gi_flooding)
                self.integral_data['gi_only_cso'].append(gi_cso)
                self.integral_data['gi_only_inflow'].append(gi_inflow)
                self.integral_data['gi_dqn_flooding'].append(self.results['flooding'][-1])
                self.integral_data['gi_dqn_cso'].append(self.results['CSO'][-1])
                self.integral_data['gi_dqn_inflow'].append(self.results['inflow'][-1])
                
                #  Calculate integrals
                dt = self.params['advance_seconds']
                
                # delta_1: Integral of inflow difference between base and GI-only
                inflow_diff = (self.integral_data['base_inflow'][-1] - 
                               self.integral_data['gi_only_inflow'][-1])
                self.integral_data['delta_1_integral'] += inflow_diff * dt
                
                # delta_2: Integral of (flooding+CSO) difference between base and GI+DQN
                base_total = (self.integral_data['base_flooding'][-1] + 
                              self.integral_data['base_cso'][-1])
                gi_dqn_total = (self.integral_data['gi_dqn_flooding'][-1] + 
                                self.integral_data['gi_dqn_cso'][-1])
                performance_diff = base_total - gi_dqn_total
                self.integral_data['delta_2_integral'] += performance_diff * dt
                
                # r1: delta_2/delta_1
                if self.integral_data['delta_1_integral'] > 0:
                    if self.integral_data['delta_2_integral'] < 0:
                        r1_raw = 0
                    else:
                        r1_raw = self.integral_data['delta_2_integral'] / self.integral_data['delta_1_integral']
                else:
                    r1_raw = 0
                
                # Map to [-1, 0]
                r1 = -np.exp(-r1_raw)
                
                # Combine rewards
                reward = 0.7 * reward3 + 0.3 * r1
            except Exception as e:
                print(f"Error in reward calculation: {e}")
                reward = reward3

        self.results['state'].append(states)
        self.results['action'].append(action)
        self.results['rewards'].append(reward)
            
        # End simulation
        if done:
            self.sim._model.swmm_end()
            self.sim._model.swmm_close()
                
            if self.baseline_sim:
                try:
                    self.baseline_sim._model.swmm_end()
                    self.baseline_sim._model.swmm_close()
                except Exception as e:
                    if "not opened" not in str(e):
                        raise
                    
            if self.gi_only_sim:
                try:
                    self.gi_only_sim._model.swmm_end()
                    self.gi_only_sim._model.swmm_close()
                except Exception as e:
                    if "not opened" not in str(e):
                        raise
                
        return states, reward, self.results, done
import os
os.environ['CONDA_DLL_SEARCH_MODIFICATION_ENABLE']="1"
import numpy as np
#import pyswmm.toolkitapi as tkai

from swmm_api.input_file import read_inp_file
from pyswmm import Simulation,Links,Nodes,RainGages,SystemStats
from swmm_api.input_file.sections.others import TimeseriesData
from swmm_api.input_file.section_labels import TIMESERIES

import matplotlib.pyplot as plt
import datetime
import yaml
import shutil 


def get_step_results(results,nodes,links,rgs,sys,config,params):
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
            
    Qtw = (sys.routing_stats['dry_weather_inflow']
                +sys.routing_stats['wet_weather_inflow']
                +sys.routing_stats['groundwater_inflow']
                +sys.routing_stats['II_inflow'])

    Qtw = sys.routing_stats['wet_weather_inflow']
    delt_Qtw = Qtw - results['inflow'][-1]
    results['inflow'].append(Qtw)
    
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

    if params['reward_type'] == '1':
        if results['inflow'][-1] == 0:
            reward = 0
        else:
            reward = - delt_flooding_time / params['advance_seconds'] - delt_cso_time / params['advance_seconds']
    elif params['reward_type'] == '2':
        if results['inflow'][-1] == 0:
            reward = 0
        else:
            reward = - (1/(results['inflow'][-1])) * ((delt_flooding * results['total_flooding_time'][-1] + delt_flooding_time * results['flooding'][-1]) 
                                        + (delt_CSO * results['total_CSO_time'][-1] + delt_cso_time * results['CSO'][-1]))
    else:
        if results['inflow'][-1] == 0:
            reward = 0
        else:
            reward = 1 / (1 + delt_flooding/delt_Qtw + delt_CSO/delt_Qtw) - 1
    return results, reward


class SWMM_ENV:
    def __init__(self,params):
        self.params = params
        self.config = yaml.load(open(self.params['parm']+".yaml"), yaml.FullLoader)
        #self.t=[]
    
    def reset(self,rain,rainid,trainlog,root):
        
        if trainlog:
            root += '/SWMM_GR/_teminp'
        else:
            root += '/SWMM_GR/_temtestinp'
        inp = read_inp_file(self.params['orf']+'.inp')
        inp[TIMESERIES]['rainfall']=TimeseriesData('rainfall',rain)
        inp.write_file(root+'/'+self.params['orf_save']+str(rainid)+'_rain.inp')
        self.sim=Simulation(root+'/'+self.params['orf_save']+str(rainid)+'_rain.inp')
        self.sim.start()
        
        if self.params['advance_seconds'] is None:
            self.sim._model.swmm_step()
        else:
            self.sim._model.swmm_stride(self.params['advance_seconds'])
                
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

        self.results = {}
        
        self.results['CSO'], self.results['flooding'], self.results['inflow'] = [0], [0], [0]
        self.results['total_flooding_time'], self.results['total_CSO_time'] = [0], [0]
        self.results['res'] = [0]
        self.results['state'], self.results['action'], self.results['rewards'] = [], [], []
        return states
        
    def step(self,action):
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
            
        for item,a in zip(self.config['action_assets'],action):
            links[item].target_setting = a
        
        if self.params['advance_seconds'] is None:
            time = self.sim._model.swmm_step()
        else:
            time = self.sim._model.swmm_stride(self.params['advance_seconds'])
        #self.t.append(self.sim._model.getCurrentSimulationTime())
        done = False if time > 0 else True

        self.results, reward = get_step_results(self.results,nodes,links,rgs,sys,self.config,self.params)

        self.results['state'].append(states)
        self.results['action'].append(action)
        self.results['rewards'].append(reward)
            
        if done:
            self.sim._model.swmm_end()
            self.sim._model.swmm_close()
        return states,reward,self.results,done
        
            

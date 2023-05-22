import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.layers import Activation
import json
import os
from keras.constraints import Constraint
import keras.backend as K 
from keras.utils.generic_utils import get_custom_objects


def mdn_cost(mu, sigma, y, indx):
    dist = tf.distributions.Normal(loc=mu, scale=sigma)
    return tf.reduce_mean(-dist.log_prob(y)*indx-dist.log_survival_function(y)*(tf.constant([[1.0]])-indx))
 
    
# constrainted NN, with relaxation
class NonPos(Constraint):
    """Constrains the weights to be non-positive.
    """
    def __call__(self, w):
        return w * K.cast(K.less_equal(w, 0.), K.floatx())

class GreaterThanMu0(Constraint):
    def __call__(self, w):
        constraint_row_0 = w[0, :] * K.cast(K.greater_equal(w[0, :], 0.), K.floatx())
        constraint_row_0 = K.expand_dims(constraint_row_0, axis=0)
        constraint_row_1 = w[1, :] * K.cast(K.greater_equal(w[1, :], 0.), K.floatx())
        constraint_row_1 = K.expand_dims(constraint_row_1, axis=0)
        constraint_row_2 = w[2, :] * K.cast(K.greater_equal(w[2, :], 0.), K.floatx())
        constraint_row_2 = K.expand_dims(constraint_row_2, axis=0)
        constraint_row_3 = w[3, :] * K.cast(K.greater_equal(w[3, :], 0.), K.floatx())
        constraint_row_3 = K.expand_dims(constraint_row_3, axis=0)
        constraint_row_4 = w[4, :] * K.cast(K.greater_equal(w[4, :], 0.), K.floatx())
        constraint_row_4 = K.expand_dims(constraint_row_4, axis=0)
        constraint_row_5 = w[5, :] * K.cast(K.greater_equal(w[5, :], 0.), K.floatx())
        constraint_row_5 = K.expand_dims(constraint_row_5, axis=0)
        constraint_row_6 = w[6, :] * K.cast(K.greater_equal(w[6, :], 0.), K.floatx())
        constraint_row_6 = K.expand_dims(constraint_row_6, axis=0)
        constraint_row_7 = w[7, :] * K.cast(K.greater_equal(w[7, :], 0.), K.floatx())
        constraint_row_7 = K.expand_dims(constraint_row_7, axis=0)
        constraint_row_8 = w[8, :] * K.cast(K.greater_equal(w[8, :], 0.), K.floatx())
        constraint_row_8 = K.expand_dims(constraint_row_8, axis=0)
        constraint_row_9 = w[9, :] * K.cast(K.greater_equal(w[9, :], 0.), K.floatx())
        constraint_row_9 = K.expand_dims(constraint_row_9, axis=0)
        full_w = K.concatenate([constraint_row_0, constraint_row_1, constraint_row_2, constraint_row_3, constraint_row_4,
                                constraint_row_5, constraint_row_6, constraint_row_7, constraint_row_8, constraint_row_9,
                                w[10:15, :]], axis=0)
        return full_w 
    
class GreaterThanSigma(Constraint):
    def __call__(self, w):
        constraint_row_0 = w[0, :] * K.cast(K.greater_equal(w[0, :], 0.), K.floatx())
        constraint_row_0 = K.expand_dims(constraint_row_0, axis=0)
        constraint_row_1 = w[1, :] * K.cast(K.greater_equal(w[1, :], 0.), K.floatx())
        constraint_row_1 = K.expand_dims(constraint_row_1, axis=0)
        constraint_row_2 = w[2, :] * K.cast(K.greater_equal(w[2, :], 0.), K.floatx())
        constraint_row_2 = K.expand_dims(constraint_row_2, axis=0)
        constraint_row_3 = w[3, :] * K.cast(K.greater_equal(w[3, :], 0.), K.floatx())
        constraint_row_3 = K.expand_dims(constraint_row_3, axis=0)
        constraint_row_4 = w[4, :] * K.cast(K.greater_equal(w[4, :], 0.), K.floatx())
        constraint_row_4 = K.expand_dims(constraint_row_4, axis=0)
        full_w = K.concatenate([constraint_row_0, constraint_row_1, constraint_row_2, constraint_row_3, constraint_row_4,
                                w[5:15, :]], axis=0)
        return full_w 
    


def activation_sigma(x):
    return K.elu(x) + 1

get_custom_objects().update({'activation_sigma': Activation(activation_sigma)})


model_name_all =['all1', 'all2', 'all3', 'all4', 'all5', 'all6', 'all7'] 
model_name_complete = ['complete_noSelNeu1', 'complete_noSelNeu2', 'complete_noSelNeu3', 'complete_noSelNeu4', 'complete_noSelNeu5', 'complete_noSelNeu6', 'complete_noSelNeu7'] # concise_model2_all_testing_500_selevtive1neuron_hidden5

history_name_all = ['history_all1', 'history_all2', 'history_all3', 'history_all4', 'history_all5', 'history_all6', 'history_all7']
history_name_complete = ['history_complete_noSelNeu1', 'history_complete_noSelNeu2', 'history_complete_noSelNeu3', 'history_complete_noSelNeu4', 'history_complete_noSelNeu5', 'history_complete_noSelNeu6', 'history_complete_noSelNeu7' ]
           
for Comp in [2,3,4,5,6,7]: 
    for all_complete in [1]: 
        for repeat_number in [1]:
            
            print(all_complete)
            print(Comp)
            print(repeat_number)
            
            AM = pd.read_csv(".../AM data fatigue.csv")
            S = AM['Sa']
            R = AM['R']
            N = AM['N']
            speed = AM['Speed']
            power = AM['Power']
            hatch = AM['Hatch']
            thickness = AM['Thickness']
            Edensity = AM['Energy density']
            temperature = AM['Heat temperature']
            time = AM['Heat time']
            pressure = AM['HIP pressure (bar)']
            machined = AM['Machined']
            polished = AM['Polished']
            complete = AM['Complete No']
            lgN = np.log10(N)
            censor_indx = AM['Censor']
            reference = AM['Ref']
            speed_missing_indx = ~speed.isnull() * 1.0
            power_missing_indx = ~power.isnull() * 1.0
            hatch_missing_indx = ~hatch.isnull() * 1.0
            thickness_missing_indx = ~thickness.isnull() * 1.0
            Edensity_missing_indx = ~Edensity.isnull() * 1.0
            temperature_missing_indx = ~temperature.isnull() * 1.0
            time_missing_indx = ~time.isnull() * 1.0
            pressure_missing_indx = ~pressure.isnull() * 1.0
            machined_missing_indx = ~machined.isnull() * 1.0
            polished_missing_indx = ~polished.isnull() * 1.0
            
            
            train_data_S = S
            train_data_R = R
            train_data_speed = speed
            train_data_power = power
            train_data_hatch = hatch
            train_data_thickness = thickness
            train_data_Edensity = Edensity
            train_data_temperature = temperature
            train_data_time = time
            train_data_pressure = pressure
            train_data_machined = machined
            train_data_polished = polished
            train_target = lgN
            
            if all_complete == 1:
                train_data_indx = reference != 0
                model_name = model_name_all
                history_name = history_name_all
            if all_complete == 2:
                train_data_indx = ~complete.isnull()
                model_name = model_name_complete
                history_name = history_name_complete
            
            _dir = ".../NN prediction randomness"      
            save_model_name = os.path.join(_dir, model_name[Comp-1]+' - %s' % repeat_number)
            save_history_name = os.path.join(_dir, history_name[Comp-1]+' - %s' % repeat_number)
            model = tf.keras.models.load_model(save_model_name, custom_objects={'NonPos': NonPos, 'GreaterThanMu0':GreaterThanMu0, 'activation_sigma':activation_sigma, 'GreaterThanSigma':GreaterThanSigma}, compile=False)
            history_dict = json.load(open(save_history_name, 'r'))
            
            plt.figure(figsize=(3, 3))
            plt.plot(history_dict['loss'])
            # plt.title('model loss')
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            # save_figure_name = os.path.join(".../Figures", history_name[Comp-1] + ' - %s' % repeat_number + '.svg')
            # plt.savefig(save_figure_name, format='svg', bbox_inches = 'tight') # ,pad_inches = 0.01
            plt.show()
    
            
            ii = S[complete == Comp].index[0]
            test_R = R[ii]
            test_speed = speed[ii]
            test_power = power[ii]
            test_hatch = hatch[ii]
            test_thickness = thickness[ii]
            test_Edensity = Edensity[ii]
            test_temperature = temperature[ii]
            test_time = time[ii]
            test_pressure = pressure[ii]
            test_machined = machined[ii]
            test_polished = polished[ii]
            
            test_data_S = np.arange(10, 1*max(S.unique()), 1)
            S_pred = test_data_S
            train_data_S_min = train_data_S.min(axis=0)
            train_data_S_max = train_data_S.max(axis=0)
            train_data_S_range = train_data_S_max - train_data_S_min
            train_data_S = (train_data_S - train_data_S_min) / train_data_S_range
            test_data_S = (test_data_S - train_data_S_min) / train_data_S_range
            
            train_data_R_min = train_data_R.min(axis=0)
            train_data_R_max = train_data_R.max(axis=0)
            train_data_R_range = train_data_R_max - train_data_R_min
            train_data_R = (train_data_R - train_data_R_min) / train_data_R_range
            test_data_R = pd.Series(np.linspace(test_R, test_R, len(test_data_S))) 
            test_data_R = (test_data_R - train_data_R_min) / train_data_R_range
            
            train_data_speed_min = train_data_speed.min(axis=0)
            train_data_speed_max = train_data_speed.max(axis=0)
            train_data_speed_range = train_data_speed_max - train_data_speed_min
            train_data_speed = (train_data_speed - train_data_speed_min) / train_data_speed_range
            test_data_speed = pd.Series(np.linspace(test_speed, test_speed, len(test_data_S))) 
            test_data_speed = (test_data_speed - train_data_speed_min) / train_data_speed_range
            
            train_data_power_min = train_data_power.min(axis=0)
            train_data_power_max = train_data_power.max(axis=0)
            train_data_power_range = train_data_power_max - train_data_power_min
            train_data_power = (train_data_power - train_data_power_min) / train_data_power_range
            test_data_power = pd.Series(np.linspace(test_power, test_power, len(test_data_S))) 
            test_data_power = (test_data_power - train_data_power_min) / train_data_power_range
            
            train_data_hatch_min = train_data_hatch.min(axis=0)
            train_data_hatch_max = train_data_hatch.max(axis=0)
            train_data_hatch_range = train_data_hatch_max - train_data_hatch_min
            train_data_hatch = (train_data_hatch - train_data_hatch_min) / train_data_hatch_range
            test_data_hatch = pd.Series(np.linspace(test_hatch, test_hatch, len(test_data_S))) 
            test_data_hatch = (test_data_hatch - train_data_hatch_min) / train_data_hatch_range
            
            train_data_thickness_min = train_data_thickness.min(axis=0)
            train_data_thickness_max = train_data_thickness.max(axis=0)
            train_data_thickness_range = train_data_thickness_max - train_data_thickness_min
            train_data_thickness = (train_data_thickness - train_data_thickness_min) / train_data_thickness_range
            test_data_thickness = pd.Series(np.linspace(test_thickness, test_thickness, len(test_data_S))) 
            test_data_thickness = (test_data_thickness - train_data_thickness_min) / train_data_thickness_range
            
            train_data_Edensity_min = train_data_Edensity.min(axis=0)
            train_data_Edensity_max = train_data_Edensity.max(axis=0)
            train_data_Edensity_range = train_data_Edensity_max - train_data_Edensity_min
            train_data_Edensity = (train_data_Edensity - train_data_Edensity_min) / train_data_Edensity_range
            test_data_Edensity = pd.Series(np.linspace(test_Edensity, test_Edensity, len(test_data_S))) 
            test_data_Edensity = (test_data_Edensity - train_data_Edensity_min) / train_data_Edensity_range
            
            train_data_temperature_min = train_data_temperature.min(axis=0)
            train_data_temperature_max = train_data_temperature.max(axis=0)
            train_data_temperature_range = train_data_temperature_max - train_data_temperature_min
            train_data_temperature = (train_data_temperature - train_data_temperature_min) / train_data_temperature_range
            test_data_temperature = pd.Series(np.linspace(test_temperature, test_temperature, len(test_data_S))) 
            test_data_temperature = (test_data_temperature - train_data_temperature_min) / train_data_temperature_range
            
            train_data_time_min = train_data_time.min(axis=0)
            train_data_time_max = train_data_time.max(axis=0)
            train_data_time_range = train_data_time_max - train_data_time_min
            train_data_time = (train_data_time - train_data_time_min) / train_data_time_range
            test_data_time = pd.Series(np.linspace(test_time, test_time, len(test_data_S))) 
            test_data_time = (test_data_time - train_data_time_min) / train_data_time_range
            
            train_data_pressure_min = train_data_pressure.min(axis=0)
            train_data_pressure_max = train_data_pressure.max(axis=0)
            train_data_pressure_range = train_data_pressure_max - train_data_pressure_min
            train_data_pressure = (train_data_pressure - train_data_pressure_min) / train_data_pressure_range
            test_data_pressure = pd.Series(np.linspace(test_pressure, test_pressure, len(test_data_S))) 
            test_data_pressure = (test_data_pressure - train_data_pressure_min) / train_data_pressure_range
            
            train_data_machined_min = train_data_machined.min(axis=0)
            train_data_machined_max = train_data_machined.max(axis=0)
            train_data_machined_range = train_data_machined_max - train_data_machined_min
            train_data_machined = (train_data_machined - train_data_machined_min) / train_data_machined_range
            test_data_machined = pd.Series(np.linspace(test_machined, test_machined, len(test_data_S))) 
            test_data_machined = (test_data_machined - train_data_machined_min) / train_data_machined_range
            
            train_data_polished_min = train_data_polished.min(axis=0)
            train_data_polished_max = train_data_polished.max(axis=0)
            train_data_polished_range = train_data_polished_max - train_data_polished_min
            train_data_polished = (train_data_polished - train_data_polished_min) / train_data_polished_range
            test_data_polished = pd.Series(np.linspace(test_polished, test_polished, len(test_data_S))) 
            test_data_polished = (test_data_polished - train_data_polished_min) / train_data_polished_range
            
            test_missing_indx = pd.Series(np.linspace(1, 1, len(test_data_S))) 
            
            
            train_data_speed[train_data_speed.isnull()] = 0
            train_data_power[train_data_power.isnull()] = 0
            train_data_hatch[train_data_hatch.isnull()] = 0
            train_data_thickness[train_data_thickness.isnull()] = 0
            train_data_Edensity[train_data_Edensity.isnull()] = 0
            train_data_temperature[train_data_temperature.isnull()] = 0
            train_data_time[train_data_time.isnull()] = 0
            train_data_pressure[train_data_pressure.isnull()] = 0
            train_data_machined[train_data_machined.isnull()] = 0
            train_data_polished[train_data_polished.isnull()] = 0
            
            
            mu_pred, sigma_pred = model.predict(list((test_data_S, test_data_R, 
                                                      test_data_speed, test_data_power, test_data_hatch, test_data_thickness, test_data_temperature, test_data_time, 
                                                      test_missing_indx, test_missing_indx, test_missing_indx, test_missing_indx, test_missing_indx, test_missing_indx, 
                                                      test_data_S, test_data_S)))
            
            plt.rcParams.update({'font.size': 11})
            plt.rcParams["font.family"] = "Times New Roman"
            fig, ax = plt.subplots(figsize=(3,3))
            axes = plt.gca()
            axes.set_xlim([2.5, 8])
            # axes.set_ylim([0.8*min(S[complete == Comp]), 1.05*max(S[complete == Comp])])
            axes.set_ylim([0, 1.05*max(S[complete == Comp])])
            plt.xlabel('log(N)')
            plt.ylabel('Sa')
            plt.scatter(lgN[(censor_indx == 1) & (R == test_R) & (speed == test_speed) & (power == test_power) & (hatch == test_hatch) & (thickness == test_thickness) & (Edensity == test_Edensity) & (temperature == test_temperature) & (time == test_time) & (pressure == test_pressure) & (machined == test_machined) & (polished == test_polished)], 
                        S[(censor_indx == 1) & (R == test_R) & (speed == test_speed) & (power == test_power) & (hatch == test_hatch) & (thickness == test_thickness) & (Edensity == test_Edensity) & (temperature == test_temperature) & (time == test_time) & (pressure == test_pressure) & (machined == test_machined) & (polished == test_polished)], 12, marker="o",c="C0", label='Failures')
            plt.scatter(lgN[(censor_indx == 0) & (R == test_R) & (speed == test_speed) & (power == test_power) & (hatch == test_hatch) & (thickness == test_thickness) & (Edensity == test_Edensity) & (temperature == test_temperature) & (time == test_time) & (pressure == test_pressure) & (machined == test_machined) & (polished == test_polished)], 
                        S[(censor_indx == 0) & (R == test_R) & (speed == test_speed) & (power == test_power) & (hatch == test_hatch) & (thickness == test_thickness) & (Edensity == test_Edensity) & (temperature == test_temperature) & (time == test_time) & (pressure == test_pressure) & (machined == test_machined) & (polished == test_polished)], 30, marker=">", edgecolors='r', facecolors='none', label='Runouts')
            plt.plot(mu_pred,S_pred,c='m', label='Mean')
            plt.plot(mu_pred-1.96*sigma_pred,S_pred,c='k',linestyle='--', label='95% CI')
            plt.plot(mu_pred+1.96*sigma_pred,S_pred,c='k',linestyle='--')
            location = ['upper right','lower left','upper right','lower left','upper right','upper right']
            plt.legend(loc=location[Comp-2],frameon=False)
            save_figure_name = os.path.join(".../Figures", model_name[Comp-1] + ' - %s' % repeat_number + '.svg')
            plt.savefig(save_figure_name, format='svg', bbox_inches = 'tight') # ,pad_inches = 0.01
            plt.show()
            
            keras.backend.clear_session()

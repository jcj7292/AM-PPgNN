import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras import optimizers
from keras.models import Model
from keras.layers import Dense, Activation, Input, Concatenate, Multiply
from scipy.stats import norm
import json
import time
import os

start = time.time()


Mean_Log_Score_all = []
Mean_Log_Score_complete = []

for all_complete in [1,2]:
    for Comp in [2,3,4,5,6,7]:
        mean_log_score_repeat = []
        for repeat_number in range(1,3):
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
            temperature = AM['Heat temperature']
            time = AM['Heat time']
            lgN = np.log10(N)
            censor_indx = AM['Censor']
            reference = AM['Ref']
            complete = AM['Complete No']
            speed_missing_indx = ~speed.isnull() * 1.0
            power_missing_indx = ~power.isnull() * 1.0
            hatch_missing_indx = ~hatch.isnull() * 1.0
            thickness_missing_indx = ~thickness.isnull() * 1.0
            temperature_missing_indx = ~temperature.isnull() * 1.0
            time_missing_indx = ~time.isnull() * 1.0
            
            train_data_S = S
            train_data_R = R
            train_data_speed = speed
            train_data_power = power
            train_data_hatch = hatch
            train_data_thickness = thickness
            train_data_temperature = temperature
            train_data_time = time
            train_target = lgN
            
            
            train_data_S_min = train_data_S.min(axis=0)
            train_data_S_max = train_data_S.max(axis=0)
            train_data_S_range = train_data_S_max - train_data_S_min
            train_data_S = (train_data_S - train_data_S_min) / train_data_S_range
            
            train_data_R_min = train_data_R.min(axis=0)
            train_data_R_max = train_data_R.max(axis=0)
            train_data_R_range = train_data_R_max - train_data_R_min
            train_data_R = (train_data_R - train_data_R_min) / train_data_R_range
            
            train_data_speed_min = train_data_speed.min(axis=0)
            train_data_speed_max = train_data_speed.max(axis=0)
            train_data_speed_range = train_data_speed_max - train_data_speed_min
            train_data_speed = (train_data_speed - train_data_speed_min) / train_data_speed_range    
            
            train_data_power_min = train_data_power.min(axis=0)
            train_data_power_max = train_data_power.max(axis=0)
            train_data_power_range = train_data_power_max - train_data_power_min
            train_data_power = (train_data_power - train_data_power_min) / train_data_power_range
              
            train_data_hatch_min = train_data_hatch.min(axis=0)
            train_data_hatch_max = train_data_hatch.max(axis=0)
            train_data_hatch_range = train_data_hatch_max - train_data_hatch_min
            train_data_hatch = (train_data_hatch - train_data_hatch_min) / train_data_hatch_range   
            
            train_data_thickness_min = train_data_thickness.min(axis=0)
            train_data_thickness_max = train_data_thickness.max(axis=0)
            train_data_thickness_range = train_data_thickness_max - train_data_thickness_min
            train_data_thickness = (train_data_thickness - train_data_thickness_min) / train_data_thickness_range
              
            
            train_data_temperature_min = train_data_temperature.min(axis=0)
            train_data_temperature_max = train_data_temperature.max(axis=0)
            train_data_temperature_range = train_data_temperature_max - train_data_temperature_min
            train_data_temperature = (train_data_temperature - train_data_temperature_min) / train_data_temperature_range  
            
            train_data_time_min = train_data_time.min(axis=0)
            train_data_time_max = train_data_time.max(axis=0)
            train_data_time_range = train_data_time_max - train_data_time_min
            train_data_time = (train_data_time - train_data_time_min) / train_data_time_range
            
            
            train_data_speed[train_data_speed.isnull()] = 0
            train_data_power[train_data_power.isnull()] = 0
            train_data_hatch[train_data_hatch.isnull()] = 0
            train_data_thickness[train_data_thickness.isnull()] = 0
            train_data_temperature[train_data_temperature.isnull()] = 0
            train_data_time[train_data_time.isnull()] = 0
            
            def mdn_cost(mu, sigma, y, indx):
                dist = tf.distributions.Normal(loc=mu, scale=sigma)
                return tf.reduce_mean(-dist.log_prob(y)*indx-dist.log_survival_function(y)*(tf.constant([[1.0]])-indx))
             
                
            # constrainted NN, with relaxation
            from keras.constraints import Constraint
            import keras.backend as K 
            
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
                
            from keras.layers import Activation
            from keras import backend as K
            from keras.utils.generic_utils import get_custom_objects
            
            def activation_sigma(x):
                return K.elu(x) + 1
            
            get_custom_objects().update({'activation_sigma': Activation(activation_sigma)})
            
              
            epochs = 1000
            batch_size = 1
            learning_rate = 0.001
            
            InputLayer_S = Input(shape=(1,))
            InputLayer_R = Input(shape=(1,))
            
            InputLayer_speed = Input(shape=(1,))
            InputLayer_power = Input(shape=(1,))
            InputLayer_hatch = Input(shape=(1,))
            InputLayer_thickness = Input(shape=(1,))
            InputLayer_temperature = Input(shape=(1,))
            InputLayer_time = Input(shape=(1,))
            InputLayer_pressure = Input(shape=(1,))
            InputLayer_polished = Input(shape=(1,))
            
            InputMask_speed = Input(shape=(1,))
            InputMask_power = Input(shape=(1,))
            InputMask_hatch = Input(shape=(1,))
            InputMask_thickness = Input(shape=(1,))
            InputMask_Edensity = Input(shape=(1,))
            InputMask_temperature = Input(shape=(1,))
            InputMask_time = Input(shape=(1,))
            
            SelectiveInputLayer_speed = Dense(1,activation="tanh")(InputLayer_speed)
            SelectiveInputLayer_power = Dense(1,activation="tanh")(InputLayer_power)
            SelectiveInputLayer_hatch = Dense(1,activation="tanh")(InputLayer_hatch)
            SelectiveInputLayer_thickness = Dense(1,activation="tanh")(InputLayer_thickness)
            SelectiveInputLayer_temperature = Dense(1,activation="tanh")(InputLayer_temperature)
            SelectiveInputLayer_time = Dense(1,activation="tanh")(InputLayer_time)
            
            multiplied_speed = Multiply()([SelectiveInputLayer_speed, InputMask_speed])
            multiplied_power = Multiply()([SelectiveInputLayer_power, InputMask_power])
            multiplied_hatch = Multiply()([SelectiveInputLayer_hatch, InputMask_hatch])
            multiplied_thickness = Multiply()([SelectiveInputLayer_thickness, InputMask_thickness])
            multiplied_temperature = Multiply()([SelectiveInputLayer_temperature, InputMask_temperature])
            multiplied_time = Multiply()([SelectiveInputLayer_time, InputMask_time])
            
            Layer_1_1 = Dense(5,activation="tanh", kernel_constraint=NonPos(), bias_constraint=NonPos())(InputLayer_S)
            Layer_1_2 = Dense(5,activation="tanh", kernel_constraint=NonPos())(InputLayer_S)
            Layer_1_3 = Dense(5,activation="tanh", kernel_constraint=NonPos())(InputLayer_R)
            Layer_1_4 = Dense(5,activation="tanh")(InputLayer_R) 
            Layer_1_5 = Dense(5,activation="tanh")(Concatenate(axis=1)([multiplied_speed, multiplied_power, multiplied_hatch, multiplied_thickness, multiplied_temperature, multiplied_time]))
            Layer_1_6 = Dense(5,activation="tanh")(Concatenate(axis=1)([multiplied_speed, multiplied_power, multiplied_hatch, multiplied_thickness, multiplied_temperature, multiplied_time]))
            merged_Layer_1_135 = Concatenate()([Layer_1_1, Layer_1_3, Layer_1_5])
            merged_Layer_1_246 = Concatenate()([Layer_1_2, Layer_1_4, Layer_1_6])
            mu0 = Dense(1, activation="linear", kernel_constraint=GreaterThanMu0())(merged_Layer_1_135)
            sigma = Dense(1, activation=activation_sigma, kernel_constraint=GreaterThanSigma())(merged_Layer_1_246)
            merged_mu0_sigma = Concatenate(axis=1)([mu0,sigma])
            mu = Dense(1, activation="linear", kernel_constraint=keras.constraints.NonNeg())(merged_mu0_sigma)
            y_real = Input(shape=(1,))
            indx = Input(shape=(1,))
            lossF = mdn_cost(mu, sigma, y_real, indx)
            model = Model(inputs=[InputLayer_S, InputLayer_R, 
                                  InputLayer_speed, InputLayer_power, InputLayer_hatch, InputLayer_thickness, InputLayer_temperature, InputLayer_time,
                                  InputMask_speed, InputMask_power, InputMask_hatch, InputMask_thickness, InputMask_temperature, InputMask_time,
                                  y_real, indx], 
                          outputs=[mu, sigma])
            model.add_loss(lossF)
            adamOptimizer = optimizers.Adam(learning_rate=learning_rate)
            model.compile(optimizer=adamOptimizer,metrics=['mse'])
            
            model_name_all =['all1', 'all2', 'all3', 'all4', 'all5', 'all6', 'all7'] 
            model_name_complete = ['complete1', 'complete2', 'complete3', 'complete4', 'complete5', 'complete6', 'complete7'] # concise_model2_all_testing_500_selevtive1neuron_hidden5
            history_name_all = ['history_all1', 'history_all2', 'history_all3', 'history_all4', 'history_all5', 'history_all6', 'history_all7']
            history_name_complete = ['history_complete1', 'history_complete2', 'history_complete3', 'history_complete4', 'history_complete5', 'history_complete6', 'history_complete7' ]
            
            
            
            if all_complete == 1:
                train_data_indx = reference != 0
                model_name = model_name_all
                history_name = history_name_all            
            if all_complete == 2:
                train_data_indx = ~complete.isnull()
                model_name = model_name_complete
                history_name = history_name_complete
      
    
        
            train_data_S = train_data_S[train_data_indx & (complete != Comp)]
            train_data_R = train_data_R[train_data_indx & (complete != Comp)]
            
            train_data_speed = train_data_speed[train_data_indx & (complete != Comp)]
            train_data_power = train_data_power[train_data_indx & (complete != Comp)]
            train_data_hatch = train_data_hatch[train_data_indx & (complete != Comp)]
            train_data_thickness = train_data_thickness[train_data_indx & (complete != Comp)]
            train_data_temperature = train_data_temperature[train_data_indx & (complete != Comp)]
            train_data_time = train_data_time[train_data_indx & (complete != Comp)]
           
            speed_missing_indx = speed_missing_indx[train_data_indx & (complete != Comp)]
            power_missing_indx = power_missing_indx[train_data_indx & (complete != Comp)]
            hatch_missing_indx = hatch_missing_indx[train_data_indx & (complete != Comp)]
            thickness_missing_indx = thickness_missing_indx[train_data_indx & (complete != Comp)]
            temperature_missing_indx = temperature_missing_indx[train_data_indx & (complete != Comp)]
            time_missing_indx =  time_missing_indx[train_data_indx & (complete != Comp)]
            
            train_target = train_target[train_data_indx & (complete != Comp)]
            train_censor_indx = censor_indx[train_data_indx & (complete != Comp)]
        
            history_cache = model.fit([train_data_S, train_data_R, 
                                        train_data_speed, train_data_power, train_data_hatch, train_data_thickness, train_data_temperature, train_data_time,
                                        speed_missing_indx, power_missing_indx, hatch_missing_indx, thickness_missing_indx, temperature_missing_indx, time_missing_indx,
                                        train_target, train_censor_indx], #notice we are using an input to pass the real values due to the inner workings of keras
                                      verbose=0, # write =1 if you wish to see the progress for each epoch
                                      epochs=epochs,
                                      batch_size=batch_size)
            print('Final cost: {0:.4f}'.format(history_cache.history['loss'][-1]))

            _dir = ".../NN prediction randomness"      
            save_model_name = os.path.join(_dir, model_name[Comp-1]+' - %s' % repeat_number)
            save_history_name = os.path.join(_dir, history_name[Comp-1]+' - %s' % repeat_number)
            model.save(save_model_name) 
            json.dump(history_cache.history, open(save_history_name, 'w'))
            
            log_score = []
            
            for ii in range(S[complete == Comp].index[0],  S[complete == Comp].index[-1]+1):
                test_S = S[complete == Comp][ii]
                test_R = R[complete == Comp][ii]
                test_speed = speed[complete == Comp][ii]
                test_power = power[complete == Comp][ii]
                test_hatch = hatch[complete == Comp][ii]
                test_thickness = thickness[complete == Comp][ii]
                test_temperature = temperature[complete == Comp][ii]
                test_time = time[complete == Comp][ii]
                test_lgN = lgN[complete == Comp][ii]
                
                test_censor_indx = censor_indx[complete == Comp][ii]
                
                test_data_S = np.linspace(test_S, test_S, 1)
                test_data_S = (test_data_S - train_data_S_min) / train_data_S_range
                test_data_R = pd.Series(np.linspace(test_R, test_R, len(test_data_S))) 
                test_data_R = (test_data_R - train_data_R_min) / train_data_R_range
                test_data_speed = pd.Series(np.linspace(test_speed, test_speed, len(test_data_S))) 
                test_data_speed = (test_data_speed - train_data_speed_min) / train_data_speed_range
                test_data_power = pd.Series(np.linspace(test_power, test_power, len(test_data_S))) 
                test_data_power = (test_data_power - train_data_power_min) / train_data_power_range
                test_data_hatch = pd.Series(np.linspace(test_hatch, test_hatch, len(test_data_S)))
                test_data_hatch = (test_data_hatch - train_data_hatch_min) / train_data_hatch_range
                test_data_thickness = pd.Series(np.linspace(test_thickness, test_thickness, len(test_data_S))) 
                test_data_thickness = (test_data_thickness - train_data_thickness_min) / train_data_thickness_range
                test_data_temperature = pd.Series(np.linspace(test_temperature, test_temperature, len(test_data_S)))
                test_data_temperature = (test_data_temperature - train_data_temperature_min) / train_data_temperature_range
                test_data_time = pd.Series(np.linspace(test_time, test_time, len(test_data_S))) 
                test_data_time = (test_data_time - train_data_time_min) / train_data_time_range
                test_missing_indx = pd.Series(np.linspace(1, 1, len(test_data_S)))
                
        
                mu_pred, sigma_pred = model.predict(list((test_data_S, test_data_R, 
                                                      test_data_speed, test_data_power, test_data_hatch, test_data_thickness, test_data_temperature, test_data_time, 
                                                      test_missing_indx, test_missing_indx, test_missing_indx, test_missing_indx, test_missing_indx, test_missing_indx,
                                                      test_data_S, test_data_S))) # the model expects a list of arrays as it has 2 inputs
                
                
                log_score.append(-np.log(norm.pdf(test_lgN, mu_pred, sigma_pred)* test_censor_indx + 
                                         (1-norm.cdf(test_lgN, mu_pred, sigma_pred))* (1-test_censor_indx))[0][0])
                
            mean_log_score_repeat.append(np.mean(log_score))
            
            keras.backend.clear_session()
        
        print(mean_log_score_repeat)
 
import time
end = time.time()
print((end - start))
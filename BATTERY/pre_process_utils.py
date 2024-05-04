
import sys
import os

import pandas as pd
import numpy as np
import h5py


from sklearn.preprocessing import MinMaxScaler
import scipy
import scipy.interpolate as interpolate


# This is the function I use to extract all of the RW discharge cycles
# I also collect the capacity values corresponding to the RW cycles
def load_curves_discharge(path,RW=True,downsample_factor=1):
    raw_data = scipy.io.loadmat(path)['data'][0][0][0][0]
    raw_data = pd.DataFrame(raw_data)
    
    
    #Asign cycle numbers to each row of data (this will segment cycles)
    cycle_num = 0
    raw_data['cycle'] = cycle_num
    raw_data['comment']=raw_data.comment.apply(lambda x: list(x)[0])
    raw_data['type'][raw_data['comment'].str.match('charge')] = 'C'
    raw_data['type']=raw_data.type.apply(lambda x: list(x)[0])
    current_type = raw_data.loc[0, 'type']
    for index in range(1, len(raw_data.index)):
        if ((current_type == "C" and raw_data.loc[index, 'type'] == "D") or 
            (current_type == "D" and raw_data.loc[index, 'type'] == "C") or
            (current_type == "R" and raw_data.loc[index, 'type'] != "R")):
            current_type = raw_data.loc[index, 'type']
            cycle_num += 1
        raw_data.loc[index, 'cycle'] = cycle_num
    
    #Select only discharge cycles (and RD cycles for capacity calculations)
    raw_data = raw_data[(raw_data.comment == 'discharge (random walk)') | (raw_data.comment == 'reference discharge')]
    output = []
    alternative_cy_count = 0
    energy = 0
    # capacity = []
    first = True
    last_cycle_capacity = 0
    RWs_seen = 0
    for jj in raw_data.cycle.unique():
        if raw_data[raw_data.cycle==jj].comment.iloc[0] == 'reference discharge':
            
            '''In here since there are usualy 2 RD cycles in a row, I will take the bigger one.. Other options also possible'''
            
            RWs_seen = 0 
            alternative_cy_count += 1
            if first:
                capacity1 = np.trapz(raw_data[raw_data.cycle==jj].current.values[0][0],raw_data[raw_data.cycle==jj].time.values[0][0])/3600
                first = False
                if capacity1>0:
                    capacity = capacity1
            else:
                capacity2 = np.trapz(raw_data[raw_data.cycle==jj].current.values[0][0],raw_data[raw_data.cycle==jj].time.values[0][0])/3600
                if abs(capacity1-last_cycle_capacity)>=abs(capacity2-last_cycle_capacity):
                    capacity = capacity2
                else:
                    capacity = capacity1
            continue
        
        if (RW) and (len(raw_data[raw_data.cycle==jj])<3): #TODO this is randomly set number to make sure each cycles has enough data points
            continue
        else:
            current_data = {}
            cy_voltage = raw_data[raw_data.cycle==jj].voltage.apply(lambda x: list(x[0])).values
            cy_voltage = np.concatenate(cy_voltage)
            cy_current = raw_data[raw_data.cycle==jj].current.apply(lambda x: list(x[0])).values
            cy_current = np.concatenate(cy_current)
            cy_temperature = raw_data[raw_data.cycle==jj].temperature.apply(lambda x: list(x[0])).values
            cy_temperature = np.concatenate(cy_temperature)
            cy_time = raw_data[raw_data.cycle==jj].time.apply(lambda x: list(x[0])).values
            cy_time = np.concatenate(cy_time)
            # cy_time = cy_time - cy_time[0]
            # cy_time = cy_time
            energy = energy + np.trapz(cy_voltage*cy_current,cy_time - cy_time[0])/3600
            # # RAW data is is every 1 sec, reduce to every 2 seconds by subsampling
            
            cy_voltage = cy_voltage[::downsample_factor]
            cy_current = cy_current[::downsample_factor]
            cy_time = cy_time[::downsample_factor]
            cy_temperature = cy_temperature[::downsample_factor]
            
            
            current_data['voltage'] = cy_voltage
            current_data['current'] = cy_current
            current_data['time'] = cy_time
            current_data['cycle'] = jj
            current_data['cycle2'] = alternative_cy_count
            current_data['capacity'] = capacity
            current_data['energy'] = energy
            current_data['temperature'] = cy_temperature
            
            first = True
            last_cycle_capacity = capacity
            RWs_seen += 1
        output.append(current_data)
    return np.array(output)


# Here I load all the data:
# Load all data split into train test
def load_all_data(dir='C:/Users/baja/Desktop/GIT_shared/prognostics/Battery/RW_Data', specific_dirs = [], train_test_split_mode = 1,downsample_factor=1):
    
    '''
    specific dirs: specificy dataset subsets that one wishes to load, otherwise loads all subsets
    train_test_split_mode: train_test_split_mode=1: first 2 batteries are train, last 2 are test
    train_test_split_mode: train_test_split_mode=2: all batteries are train, none are test (can do the split later as I will demonstrate)
    downsample_factor: integer, downsample factor for the data, 1 means no downsampling

    
    '''
    V_train = []
    I_train = []
    Temperature_train = []
    C_train = []
    C_train_2 = []
    U_train = []
    HI_train = []
    Energy_train = []
    Time_train = []
    
    V_test = []
    I_test = []
    Temperature_test = []
    C_test = []
    C_test_2 = []
    U_test = []
    HI_test = []
    Energy_test = []
    Time_test = []
    
    battery_train_names = []
    battery_test_names = []
    
    paths = os.listdir(dir)
    
    if len(specific_dirs) > 0:
        idx = np.isin(paths,specific_dirs)
        paths = np.array(paths)[idx]
    for path in paths:
        batteries = os.listdir(dir+path + '/data/Matlab/')
        print('LOADING DATA FROM: ' + path)
        if train_test_split_mode == 1:
            batteries_train = batteries[:2]
            batteries_test = batteries[2:]
            
            battery_train_names.extend(batteries_train)
            battery_test_names.extend(batteries_test)
        elif train_test_split_mode == 2:
            batteries_train = batteries
            batteries_test = []
            battery_train_names.extend(batteries_train)
            
            
        print('TRAIN BATTERIES: ' + str(batteries_train))
        print('TEST BATTERIES: ' + str(batteries_test))
        for c in range(len(batteries_train)):
            curves = load_curves_discharge(path = dir + path + '/data/Matlab/'+batteries_train[c],downsample_factor=downsample_factor)
            print(np.max([len(curves[jj]['voltage']) for jj in range(len(curves))]))
            for jj in range(len(curves)):
                V_train.extend(curves[jj]['voltage'])
                I_train.extend(curves[jj]['current'])
                Temperature_train.extend(curves[jj]['temperature'])
                C_train.extend(np.ones(len(curves[jj]['voltage']))*jj)
                C_train_2.extend(np.ones(len(curves[jj]['voltage']))*curves[jj]['cycle2'])
                U_train.extend(np.ones(len(curves[jj]['voltage']))*(int(batteries_train[c][2:-4])))
                HI_train.extend(np.ones(len(curves[jj]['voltage']))*curves[jj]['capacity'])
                Energy_train.extend(np.ones(len(curves[jj]['voltage']))*curves[jj]['energy'])
                Time_train.extend(curves[jj]['time'])
                
        for c in range(len(batteries_test)):
            curves = load_curves_discharge(path = dir + path + '/data/Matlab/'+batteries_test[c],downsample_factor=downsample_factor)
            print(np.max([len(curves[jj]['voltage']) for jj in range(len(curves))]))
            for jj in range(len(curves)):
                V_test.extend(curves[jj]['voltage'])
                I_test.extend(curves[jj]['current'])
                Temperature_test.extend(curves[jj]['temperature'])
                C_test.extend(np.ones(len(curves[jj]['voltage']))*jj)
                C_test_2.extend(np.ones(len(curves[jj]['voltage']))*curves[jj]['cycle2'])
                U_test.extend(np.ones(len(curves[jj]['voltage']))*(-1+len(battery_test_names)+c))
                HI_test.extend(np.ones(len(curves[jj]['voltage']))*curves[jj]['capacity'])
                Energy_test.extend(np.ones(len(curves[jj]['voltage']))*curves[jj]['energy'])
                Time_test.extend(curves[jj]['time'])
                
                
                
                
    V_train=np.array(V_train).astype(np.float16)
    I_train = np.array(I_train).astype(np.float16)
    C_train=np.array(C_train).astype(np.float16)
    C_train_2=np.array(C_train_2).astype(np.float16)
    U_train=np.array(U_train).astype(np.float16)
    HI_train=np.array(HI_train).astype(np.float16)
    Energy_train=np.array(Energy_train).astype(np.float16)
    Time_train=np.array(Time_train)
    Temperature_train = np.array(Temperature_train).astype(np.float16)
    
    V_test=np.array(V_test).astype(np.float16)
    I_test = np.array(I_test).astype(np.float16)
    C_test=np.array(C_test).astype(np.float16) 
    C_test_2=np.array(C_test_2).astype(np.float16)
    U_test=np.array(U_test).astype(np.float16)
    HI_test=np.array(HI_test).astype(np.float16)
    Energy_test=np.array(Energy_test).astype(np.float16)
    Time_test = np.array(Time_test)
    Temperature_test = np.array(Temperature_test).astype(np.float16)
    
    return V_train,I_train,Temperature_train,C_train,C_train_2,U_train,HI_train, Energy_train, Time_train ,\
V_test,I_test,Temperature_test,C_test,C_test_2,U_test,HI_test, Energy_test, Time_test, \
battery_train_names,battery_test_names



def HI_interpolator(HI,units,cycles,cycles2):
    HI_out = []
    for j in np.unique(units):
        real_c_id = []
        real_hi_id = [] 
        idx = np.ravel(units==j)
        HI_unit = HI[idx]
        cycles_unit = cycles[idx]
        cycles_unit2 = cycles2[idx]
        for i in np.unique(cycles_unit2):
            idx2 = np.ravel(cycles_unit2==i)
            # print(cycles_unit[idx2][0])
            real_c_id.append(cycles_unit[idx2][0][0])
            real_hi_id.append(HI_unit[idx2][0][0])
        interp = np.interp(cycles_unit, real_c_id, real_hi_id).reshape(-1,1)
        HI_out.append(interp)
    return np.vstack(HI_out)

def RUL_cut_all(V,I,T,U,C,C2,HI,Time,Capacity_cut,nomi_cap):
    V_out = []
    I_out = []
    T_out = []
    U_out = []
    C_out = []
    C2_out = []
    HI_out = []   
    Time_out = []
    for jj in np.unique(U):
        idx = np.ravel(U == jj)
        print(jj,np.min(HI[idx]/nomi_cap))
        if np.min(HI[idx]/nomi_cap) > Capacity_cut:
            cut = len(HI[idx])
        else:
            cut = np.where(HI[idx]/nomi_cap < Capacity_cut)[0][0]+1
        V_out.extend(V[idx][:cut])
        I_out.extend(I[idx][:cut])
        T_out.extend(T[idx][:cut])
        U_out.extend(U[idx][:cut])
        C_out.extend(C[idx][:cut])
        C2_out.extend(C2[idx][:cut])
        HI_out.extend(HI[idx][:cut])
        Time_out.extend(Time[idx][:cut])
    return np.array(V_out), np.array(I_out), np.array(T_out), np.array(U_out), np.array(C_out), np.array(C2_out), np.array(HI_out), np.array(Time_out)


def RUL_maker(C,U):
    RUL_train = []
    indexes = np.unique(U,return_index=True)[1]
    # print([U[index] for index in sorted(indexes)])
    for jj in [U[index] for index in sorted(indexes)]: # For each unique unit in order of appearace
        idx = np.ravel(U == jj)
        cycles = C[idx]
        mx = cycles.max()
        for cc in np.unique(cycles): # For each cycle
            RUL_train.extend((np.ones(len(cycles[cycles == cc]))*(mx-cc)))
    RUL_train = np.array(RUL_train) 
    return RUL_train


def sequence_generator_full_trajectory(x,unit,cycle,max = 0):
    x_out = []
    c_out = []
    u_out = []
    
    if max == 0:
        for jj in np.unique(unit):
            idx = np.ravel(unit==jj)
            cs = cycle[idx]
            max_len_cycle = np.max([len(cs[cs==cc]) for cc in np.unique(cs)])
            if max_len_cycle > max:
                max = max_len_cycle
    
    print(max)
    
    for jj in np.unique(unit):
        idx = np.ravel(unit==jj)
        x_ = x[idx]
        c_ = cycle[idx]
        for cc in np.unique(c_):
            idx_ = np.ravel(c_==cc)
            if len(x_[idx_]) <= max:
                x_out.append(np.concatenate([x_[idx_],np.zeros((max-len(x_[idx_]),1))]))
            else:
                x_out.append(x_[idx_][:max])
            c_out.append(cc)
            u_out.append(jj)
    
    return np.array(x_out),np.array(c_out),np.array(u_out)


# Function to sequence the data with a sliding window
def split_sequences(input_data, sequence_length, stride = 1, option = None):
    """
     
    """
    X = list()
    
    for i in range(0,len(input_data),stride):
        # find the end of this pattern
        end_ix = i + sequence_length
        
        # check if we are beyond the dataset
        if end_ix > len(input_data):
            break
        
        # gather input and output parts of the pattern
        if option=='last':
            seq_x = input_data[end_ix-1, :]
        elif option=='next':
            seq_x = input_data[end_ix, :]
        else:
            seq_x = input_data[i:end_ix]
        X.append(seq_x)
    
    return np.array(X)

def sequence_generator(input_data, cycles,units, sequence_length=10, stride =1,option=None):
    """
     # Generates dataset with windows of sequence_length      
    """  
    X = list()
    u_num=[]
    c_num =[]
    for i in np.unique(units):
        idx = np.ravel(units==i)
        x_unit = input_data[idx]
        c_unit = cycles[idx]
        for j in np.unique(c_unit):
            mask = np.ravel(c_unit==j)
            seq_x_u = split_sequences(x_unit[mask],sequence_length, stride ,option)
            X.append(seq_x_u)
            c_num.extend(np.ones(len(seq_x_u),dtype = int)*j)
            u_num.extend(np.ones(len(seq_x_u),dtype = int)*i)
    
    return np.vstack(X),np.array(c_num).reshape(-1,1), np.array(u_num).reshape(-1,1)
import numpy as np


def sample_data(xs,w,y,t,units,cycles,hi,freq,mean = False):
    X_out,W_out,Y_out,T_out,U_out,C_out,HI_out= [], [], [], [], [], [] , []
    uniq_units=np.unique(units)
    for k in uniq_units:
        unit = np.ravel(units == k)
        u_unit=units[unit,:]
        w_unit = w[unit,:]
        x_s_unit = xs[unit,:]
        y_unit= y[unit,:]
        t_unit= t[unit,:]
        c_unit= cycles[unit,:]
        h_unit = hi[unit]
        
        if mean:
            for i in range(0,x_s_unit.shape[0],freq):
                X_out.append(np.mean(x_s_unit[i:i+freq,:],axis=0))
                W_out.append(np.mean(w_unit[i:i+freq,:],axis=0))
                Y_out.append(np.mean(y_unit[i:i+freq,:],axis=0))
                T_out.append(np.mean(t_unit[i:i+freq,:],axis=0))
                HI_out.append(np.mean(h_unit[i:i+freq,:],axis=0))
            
            C_out.extend(c_unit[::freq])
            U_out.extend(u_unit[::freq])
        else:
        
            X_out.extend(x_s_unit[::freq,])
            W_out.extend(w_unit[::freq,])
            Y_out.extend(y_unit[::freq])
            U_out.extend(u_unit[::freq])
            T_out.extend(t_unit[::freq])
            C_out.extend(c_unit[::freq])
            HI_out.extend(h_unit[::freq])
            
    return  np.array(X_out).astype(np.float16),\
np.array(W_out).astype(np.float16),\
np.array(Y_out).astype(np.float16),\
np.array(T_out).astype(np.float16),\
np.array(U_out).astype(np.float16),\
np.array(C_out).astype(np.float16),\
np.array(HI_out).astype(np.float16)


# np.array(X_out), np.array(W_out), np.array(Y_out), np.array(T_out), np.array(U_out), np.array(C_out), np.array(HI_out)





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
            seq_x = input_data[i:end_ix, :]
        X.append(seq_x)
    
    return np.array(X)



def sequence_generator(input_data, units, cycles, sequence_length=10,stride = 1, option=None):
    """
     # Generates dataset with windows of sequence_length      
    """  
    X = list()
    unit_num=[]
    c_num =[]
    for i, elem_u in enumerate(list(np.unique(units))):
        mask = np.ravel(units==elem_u)
        c_mask = cycles[mask]
        x_unit = input_data[mask]
        for j in np.unique(c_mask):
            mask = np.ravel(c_mask==j)
            seq_x_u = split_sequences(x_unit[mask],sequence_length, stride, option)
            X.append(seq_x_u)
            unit_num.extend(np.ones(len(seq_x_u),dtype = int)*elem_u)
            c_num.extend(np.ones(len(seq_x_u),dtype = int)*j)
    
    return np.vstack(X),np.array(unit_num).reshape(-1,1),np.array(c_num).reshape(-1,1)


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
                x_out.append(np.concatenate([x_[idx_],np.zeros((max-len(x_[idx_]),x_.shape[-1]))]))
            else:
                x_out.append(x_[idx_][:max])
            c_out.append(cc)
            u_out.append(jj)
    
    return np.array(x_out),np.array(c_out),np.array(u_out)





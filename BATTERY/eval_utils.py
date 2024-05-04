
import numpy as np
import time

# EVALUATION modules
from scipy.stats import spearmanr
from sklearn import feature_selection
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler


import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

def init_rul_result_dict():
    
    results = {
        'MAE_TR': [],
        'MAPE_TR': [],
        'RMSE_TR': [],
        'MAE_TS': [],
        'MAPE_TS': [],
        'RMSE_TS': [],
        'TIME': [],
        'I': []
    }

    return results


def init_hi_result_dict():
    
    results = {
        'MAE_TR': [],
        'MAPE_TR': [],
        'M_TR': [],
        'T_TR': [],
        'FC_TR': [],
        'MI_TR': [],
        'MAE_TS': [],
        'MAPE_TS': [],
        'M_TS': [],
        'T_TS': [],
        'FC_TS': [],
        'MI_TS': [],
        'TIME': [],
        'I': []
    }

    return results


def expand_hi(cycles_real,units_real,HI,cyles_hi,units_hi):
    result = []
    for i in np.unique(units_real):
        idx = np.ravel(units_real==i)
        idx2 = np.ravel(units_hi==i)
        for j in np.unique(cycles_real[idx]):
            idx_cycle = np.ravel(cycles_real[idx]==j)
            idx_cycle2 = np.ravel(cyles_hi[idx2]==j)
            result.extend(np.ones(len(cycles_real[idx][idx_cycle]))*HI[idx2][idx_cycle2])
    result = np.array(result)
    return np.concatenate(result).reshape(-1,1)


def smooth_per_cycle(z,cycles,units):
    result = []
    units_out = []
    cycles_out = []
    for j in np.unique(units):
        idx = np.ravel(units==j)
        zz = z[idx]
        for i in np.unique(cycles[idx]):
            result.append(np.mean(zz[cycles[idx]==i],axis=0))
            units_out.append(j)
            cycles_out.append(i)
    return np.array(result), np.array(units_out),np.array(cycles_out)


def smooth_per_cycle_2(z,cycles,units):
    result = []
    result_std = []
    units_out = []
    cycles_out = []
    for j in np.unique(units):
        idx = np.ravel(units==j)
        zz = z[idx]
        for i in np.unique(cycles[idx]):
            result.append(np.mean(zz[cycles[idx]==i],axis=0))
            result_std.append(np.std(zz[cycles[idx]==i],axis=0))
            units_out.append(j)
            cycles_out.append(i)
    return np.array(result),np.array(result_std), np.array(units_out),np.array(cycles_out)


def rescale(x,a,b,x_min,x_max):
    '''rescale x from [x_min,x_
    max] to [a,b]  '''
    return (b-a)*(x-x_min)/(x_max-x_min) + a


def train_and_evaluate_residual_model(model_type,X_train, W_train, C_train,C_train_2, U_train, Y_train, H_eval,
                             X_test, W_test, C_test,C_test_2, U_test, Y_test, H_eval_test,
                             model,
                             healthy_thresholds=[20], runs=1, epochs=20, batch_size=512,
                             learning_rate=0.0001, option='pca',reset_weights=True):
    
    results = init_hi_result_dict()
    z_train_history = []
    z_std_train_history = []
    z_test_history = []
    z_std_test_history = []
    initial_weights = model.get_weights()

    
    for threshold in healthy_thresholds:
        for j in range(runs):
            print('RUN#: ',j)
            start_time = time.time()
            if reset_weights:
                print('reset_weights')
                model.set_weights(initial_weights)

            # Define model
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
            loss = tf.keras.losses.MeanAbsoluteError()
            model.compile(optimizer=optimizer, loss=loss)

            idx = np.ravel(C_train <= threshold)
            X_healthy = X_train[idx]
            W_healthy = W_train[idx]
            
            if model_type == 'A':                    
                train_ds = tf.data.Dataset.from_tensor_slices(((X_healthy, W_healthy), X_healthy))
            elif model_type == 'B':
                train_ds = tf.data.Dataset.from_tensor_slices((W_healthy, X_healthy))
            else: 
                raise ValueError("Invalid model type. Must be 'A',or 'B'")
                
            train_ds = train_ds.shuffle(buffer_size=X_healthy.shape[0]).batch(batch_size)
            history = model.fit(train_ds, epochs=epochs, verbose=0)


            if model_type == 'A':   
                pred_train = model.predict([X_train, W_train], batch_size=batch_size)
            if model_type == 'B':   
                pred_train = model.predict(W_train, batch_size=batch_size)
                
            resid_train = np.abs(pred_train - X_train)
            
            # Dimensionality reduction
            if option == 'pca':
                resid_train = resid_train.reshape(resid_train.shape[0], -1)
                pca = PCA(n_components=1)
                z = pca.fit_transform(resid_train)
            else:
                resid_train = np.mean(resid_train, axis=1)
                z = np.sum(resid_train, axis=1).reshape(-1, 1)


                
            zz_scaled,zz_std,units_zz,cycles_zz = smooth_per_cycle_2(z,C_train_2[:,0],U_train)
            hh = smooth_per_cycle(H_eval.reshape(-1,1),C_train_2[:,0],U_train)[0]
            yz = smooth_per_cycle(Y_train,C_train_2[:,0],U_train)[0]
            
            res = []
            for i in np.unique(units_zz):
                idx = np.ravel(units_zz==i)
                hi = zz_scaled[idx]
                res.append(np.mean(hi[0:10]-hi[-10:]))
                increasing = np.mean(res)<0

            if increasing:
                zz_scaled = -zz_scaled

                
            max_capactity = np.max(hh)
            min_capacity = np.min(hh)
            max_z = np.max(zz_scaled)
            min_z = np.min(zz_scaled)
            zz_scaled = rescale(zz_scaled,min_capacity,max_capactity,min_z,max_z)
            zz_std_scaled = zz_std*(max_capactity-min_capacity)/(max_z-min_z)
            
            zz_scaled = zz_scaled * 100
            hh = hh*100
            zz_std_scaled = zz_std_scaled*100
            
            results['MAPE_TR'].append(mape(hh[hh!=0],zz_scaled[hh!=0]))
            results['MAE_TR'].append(mae(hh,zz_scaled))

            results['M_TR'].append(monotonicity(zz_scaled,units_zz))
            results['T_TR'].append(trend_spearman(zz_scaled,cycles_zz,units_zz))
            results['FC_TR'].append(failure_consistency(zz_scaled,units_zz))
            results['MI_TR'].append(mutual_information(zz_scaled.reshape(-1,1),yz,units_zz))
            
            z_train_history.append(zz_scaled)
            z_std_train_history.append(zz_std_scaled)
            
            '''
            --------------------------------------------------------------------------------
            '''
            
            # Testing  X_test, W_test, C_test, U_test, Y_test, 
            if model_type == 'A':   
                pred_test = model.predict([X_test, W_test], batch_size=batch_size)
            if model_type == 'B':   
                pred_test = model.predict(W_test, batch_size=batch_size)
            
            
            resid_test = np.abs(pred_test - X_test)
            
            # Dimensionality reduction
            if option == 'pca':
                resid_test = resid_test.reshape(resid_test.shape[0], -1)
                z = pca.transform(resid_test)
            else:
                resid_test = np.mean(resid_test, axis=1)
                z = np.sum(resid_test, axis=1).reshape(-1, 1)
            
            
            zz_scaled,zz_std,units_zz,cycles_zz = smooth_per_cycle_2(z,C_test_2[:,0],U_test)
            hh = smooth_per_cycle(H_eval_test.reshape(-1,1),C_test_2[:,0],U_test)[0]
            yz = smooth_per_cycle(Y_test,C_test_2[:,0],U_test)[0]
            
            
            if increasing:
                zz_scaled = -zz_scaled
            
            zz_scaled = rescale(zz_scaled,min_capacity,max_capactity,min_z,max_z)
            zz_std_scaled = zz_std*(max_capactity-min_capacity)/(max_z-min_z)
            
            zz_scaled = zz_scaled * 100
            hh = hh*100
            zz_std_scaled = zz_std_scaled*100
            
            
            results['MAPE_TS'].append(mape(hh[hh!=0],zz_scaled[hh!=0]))
            results['MAE_TS'].append(mae(hh,zz_scaled))

            results['M_TS'].append(monotonicity(zz_scaled,units_zz))
            results['T_TS'].append(trend_spearman(zz_scaled,cycles_zz,units_zz))
            results['FC_TS'].append(failure_consistency(zz_scaled,units_zz))
            results['MI_TS'].append(mutual_information(zz_scaled.reshape(-1,1),yz,units_zz))
            
            z_test_history.append(zz_scaled)
            z_std_test_history.append(zz_std_scaled)
            
            
            elapsed_time = time.time() - start_time
            
            # Update results dictionary
            results['TIME'].append(elapsed_time)
            results['I'].append(threshold)
            
            print(results)
    return results,np.array(z_train_history),np.array(z_std_train_history),np.array(z_test_history),np.array(z_std_test_history)



def train_and_evaluate_supervised_model(X_train, W_train, C_train,C_train_2, U_train, Y_train,hi_train,H_eval,
                             X_test, W_test, C_test,C_test_2, U_test, Y_test, hi_test,H_eval_test,
                             model,
                             runs=1, epochs=20, batch_size=512,
                             learning_rate=0.0001,reset_weights = True):
    
    results = init_hi_result_dict()
    z_train_history = []
    z_std_train_history = []
    z_test_history = []
    z_std_test_history = []
    initial_weights = model.get_weights()
    

    for j in range(runs):
        print('RUN#: ',j)
        start_time = time.time()
        
        if reset_weights:
            print('reset_weights')
            model.set_weights(initial_weights)

        # Define model
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        loss = tf.keras.losses.MeanAbsoluteError()
        model.compile(optimizer=optimizer, loss=loss)

                
        train_ds = tf.data.Dataset.from_tensor_slices(((X_train, W_train), hi_train))
        train_ds = train_ds.shuffle(buffer_size=X_train.shape[0]).batch(batch_size)
        history = model.fit(train_ds, epochs=epochs, verbose=1)

  
        z = model.predict([X_train, W_train], batch_size=batch_size)

            

        zz_scaled,zz_std,units_zz,cycles_zz = smooth_per_cycle_2(z,C_train_2[:,0],U_train)
        hh = smooth_per_cycle(H_eval.reshape(-1,1),C_train_2[:,0],U_train)[0]
        yz = smooth_per_cycle(Y_train,C_train_2[:,0],U_train)[0]

            
        max_capactity = np.max(hh)
        min_capacity = np.min(hh)
        max_z = np.max(zz_scaled)
        min_z = np.min(zz_scaled)
        zz_scaled = rescale(zz_scaled,min_capacity,max_capactity,min_z,max_z)
        zz_std_scaled = zz_std*(max_capactity-min_capacity)/(max_z-min_z)
        
        zz_scaled = zz_scaled * 100
        hh = hh*100
        zz_std_scaled = zz_std_scaled*100
        plt.plot(zz_scaled)
        plt.plot(hh)
        plt.show()
        
        
        results['MAPE_TR'].append(mape(hh[hh!=0],zz_scaled[hh!=0]))
        results['MAE_TR'].append(mae(hh,zz_scaled))

        results['M_TR'].append(monotonicity(zz_scaled,units_zz))
        results['T_TR'].append(trend_spearman(zz_scaled,cycles_zz,units_zz))
        results['FC_TR'].append(failure_consistency(zz_scaled,units_zz))
        results['MI_TR'].append(mutual_information(zz_scaled.reshape(-1,1),yz,units_zz))
        
        z_train_history.append(zz_scaled)
        z_std_train_history.append(zz_std_scaled)
        
        '''
        --------------------------------------------------------------------------------
        '''
        
        # Testing  X_test, W_test, C_test, U_test, Y_test, 

        z = model.predict([X_test, W_test], batch_size=batch_size)

        zz_scaled,zz_std,units_zz,cycles_zz = smooth_per_cycle_2(z,C_test_2[:,0],U_test)
        hh = smooth_per_cycle(H_eval_test.reshape(-1,1),C_test_2[:,0],U_test)[0]
        yz = smooth_per_cycle(Y_test,C_test_2[:,0],U_test)[0]
        
        zz_scaled = rescale(zz_scaled,min_capacity,max_capactity,min_z,max_z)
        zz_std_scaled = zz_std*(max_capactity-min_capacity)/(max_z-min_z)
        
        zz_scaled = zz_scaled * 100
        hh = hh*100
        zz_std_scaled = zz_std_scaled*100
        
        
        results['MAPE_TS'].append(mape(hh[hh!=0],zz_scaled[hh!=0]))
        results['MAE_TS'].append(mae(hh,zz_scaled))

        results['M_TS'].append(monotonicity(zz_scaled,units_zz))
        results['T_TS'].append(trend_spearman(zz_scaled,cycles_zz,units_zz))
        results['FC_TS'].append(failure_consistency(zz_scaled,units_zz))
        results['MI_TS'].append(mutual_information(zz_scaled.reshape(-1,1),yz,units_zz))
        
        z_test_history.append(zz_scaled)
        z_std_test_history.append(zz_std_scaled)
        
        
        elapsed_time = time.time() - start_time
        
        # Update results dictionary
        results['TIME'].append(elapsed_time)
        results['I'].append(j)
        
        print(results)
    return results,np.array(z_train_history),np.array(z_std_train_history),np.array(z_test_history),np.array(z_std_test_history)




def train_and_evaluate_proposed_model(X_windows, W_windows, C_windows,C_windows_2, U_windows, Y_windows,H_eval,
                                      X_windows_test,W_windows_test,C_windows_test,C_windows_test_2, U_windows_test,Y_windows_test,H_eval_test,
                                        model,
                                        runs=1, epochs=20, batch_size=20,
                                        learning_rate=0.0001,shuffle = True, reset_weights = True):
    results = init_hi_result_dict()
    initial_weights = model.get_weights()
    z_train_history = []
    z_std_train_history = []
    z_test_history = []
    z_std_test_history = []

    train_ds = tf.data.Dataset.from_tensor_slices(((X_windows, C_windows.reshape(-1, 1), W_windows), X_windows))
    
    if shuffle:
        train_ds = train_ds.shuffle(buffer_size=X_windows.shape[0]).batch(batch_size)
    else:
        train_ds = train_ds.batch(batch_size).shuffle(buffer_size=X_windows.shape[0])
    for j in range(runs):
        print('RUN#: ',j)
        start_time = time.time()
        print('reset_weights')
        if reset_weights:
            model.set_weights(initial_weights)
            
        # MODEL
        WINDOW_LEN = X_windows.shape[1]
        OPTIMIZER = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        LOSS = tf.keras.losses.MeanAbsoluteError()
        model.compile(optimizer=OPTIMIZER, loss=LOSS, metrics=['mae'])

        history = model.fit(train_ds, epochs=epochs, verbose=1)

        laten_extract = tf.keras.models.Model(inputs=model.inputs[0],
                                               outputs=[model.get_layer("Z").output])

        z = laten_extract.predict(X_windows, batch_size=batch_size)
        
        z,z_std, units_zz,cycles_zz = smooth_per_cycle_2(z,C_windows_2[:,0],U_windows)
        hh = smooth_per_cycle(H_eval,C_windows_2[:,0],U_windows)[0]
        yz = smooth_per_cycle(Y_windows,C_windows_2[:,0],U_windows)[0]

        max_capactity = np.max(hh)
        min_capacity = np.min(hh)
        max_z = np.max(z)
        min_z = np.min(z)
        zz_scaled = rescale(z,min_capacity,max_capactity,min_z,max_z)
        z_std = z_std * (max_capactity-min_capacity)/ (max_z-min_z)
        
        zz_scaled = zz_scaled*100
        hh = hh*100
        z_std = z_std * 100
        
        zz_scaled = zz_scaled.ravel()
        hh = hh.ravel()
        
        results['MAPE_TR'].append(mape(hh[hh!=0],zz_scaled[hh!=0]))
        results['MAE_TR'].append(mae(hh,zz_scaled))

        results['M_TR'].append(monotonicity(z,units_zz))
        results['T_TR'].append(trend_spearman(z,cycles_zz,units_zz))
        results['FC_TR'].append(failure_consistency(z,units_zz))
        results['MI_TR'].append(mutual_information(z.reshape(-1,1),yz.reshape(-1,1),units_zz))
        
        z_train_history.append(zz_scaled)
        z_std_train_history.append(z_std)
        
        
        # TEST
        z = laten_extract.predict(X_windows_test, batch_size=batch_size)
        
        z,z_std, units_zz,cycles_zz = smooth_per_cycle_2(z,C_windows_test_2[:,0],U_windows_test)
        hh = smooth_per_cycle(H_eval_test,C_windows_test_2[:,0],U_windows_test)[0]
        yz = smooth_per_cycle(Y_windows_test,C_windows_test_2[:,0],U_windows_test)[0]
        
        zz_scaled = rescale(z,min_capacity,max_capactity,min_z,max_z)
        z_std = z_std * (max_capactity-min_capacity)/ (max_z-min_z)
        zz_scaled = zz_scaled*100
        hh = hh*100
        z_std = z_std * 100
        zz_scaled = zz_scaled.ravel()
        hh = hh.ravel()
        results['MAPE_TS'].append(mape(hh[hh!=0],zz_scaled[hh!=0]))
        results['MAE_TS'].append(mae(hh,zz_scaled))

        results['M_TS'].append(monotonicity(z,units_zz))
        results['T_TS'].append(trend_spearman(z,cycles_zz,units_zz))
        results['FC_TS'].append(failure_consistency(z,units_zz))
        results['MI_TS'].append(mutual_information(z.reshape(-1,1),yz.reshape(-1,1),units_zz))
        
        z_test_history.append(zz_scaled)
        z_std_test_history.append(z_std)
        elapsed_time = time.time() - start_time
        results['TIME'].append(elapsed_time)
        results['I'].append(j)
        print(results, flush=True)
    
    return results,np.array(z_train_history),np.array(z_std_train_history),np.array(z_test_history),np.array(z_std_test_history)


def train_and_evaluate_rul_model(X_windows, W_windows, C_windows,C_windows_2, U_windows, Y_windows,hi_train,
                                      X_windows_test,W_windows_test,C_windows_test, C_windows_test_2,U_windows_test,Y_windows_test,hi_test,
                                        model,useH = False,
                                        runs=1, epochs=20, batch_size=20,
                                        learning_rate=0.0001, reset_weights = True):
    results = init_rul_result_dict()
    initial_weights = model.get_weights()
    seed = 229
    np.random.seed(seed)
    shuffle_idx = np.random.permutation(X_windows.shape[0])
    training_idx, val_idx = shuffle_idx[X_windows.shape[0]//10:],shuffle_idx[:X_windows.shape[0]//10]
    if not useH:
        train_ds = tf.data.Dataset.from_tensor_slices(((X_windows[training_idx],W_windows[training_idx],C_windows[training_idx]),Y_windows[training_idx]))
        train_ds = train_ds.shuffle(buffer_size=X_windows.shape[0]).batch(batch_size)     
        val_ds = tf.data.Dataset.from_tensor_slices(((X_windows[val_idx],W_windows[val_idx],C_windows[val_idx]),Y_windows[val_idx]))    
        val_ds = val_ds.shuffle(buffer_size=X_windows.shape[0]).batch(batch_size)

        test_ds = tf.data.Dataset.from_tensor_slices(((X_windows_test,W_windows_test,C_windows_test),Y_windows_test))
        test_ds = test_ds.batch(batch_size)
    else:
        train_ds = tf.data.Dataset.from_tensor_slices(((X_windows[training_idx],W_windows[training_idx],C_windows[training_idx],hi_train[training_idx]),Y_windows[training_idx]))
        train_ds = train_ds.shuffle(buffer_size=X_windows.shape[0]).batch(batch_size)     
        val_ds = tf.data.Dataset.from_tensor_slices(((X_windows[val_idx],W_windows[val_idx],C_windows[val_idx],hi_train[val_idx]),Y_windows[val_idx]))    
        val_ds = val_ds.shuffle(buffer_size=X_windows.shape[0]).batch(batch_size)

        test_ds = tf.data.Dataset.from_tensor_slices(((X_windows_test,W_windows_test,C_windows_test,hi_test),Y_windows_test))
        test_ds = test_ds.batch(batch_size) 

    for j in range(runs):
        print('RUN#: ',j)
        start_time = time.time()
        if reset_weights:
            print('reset_weights')
            model.set_weights(initial_weights)
            
        # MODEL
        OPTIMIZER = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        LOSS = tf.keras.losses.MeanAbsoluteError()
        model.compile(optimizer=OPTIMIZER, loss=LOSS)

        callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5,restore_best_weights=True)  
        history = model.fit(train_ds,epochs=epochs,validation_data=val_ds,verbose=1,callbacks = [callback])


        #TRAIN RESULTS
        y_temp = Y_windows[val_idx]
        pred = model.predict(val_ds)
        
        pred= smooth_per_cycle_2(pred,C_windows_2[val_idx],U_windows[val_idx])[0]
        y_temp= smooth_per_cycle(y_temp,C_windows_2[val_idx],U_windows[val_idx])[0]
        pred = pred.reshape(-1,1)
        y_temp = y_temp.reshape(-1,1)
        
        
        results['MAPE_TR'].append(mape(y_temp[y_temp!=0],pred[y_temp!=0]))
        results['MAE_TR'].append(mae(y_temp,pred))
        results['RMSE_TR'].append(np.sqrt(np.mean((pred-y_temp)**2)))

        
        #TEST RESULTS
        y_temp = Y_windows_test
        pred = model.predict(test_ds)
        pred = smooth_per_cycle_2(pred,C_windows_test_2,U_windows_test)[0]
        y_temp = smooth_per_cycle(y_temp,C_windows_test_2,U_windows_test)[0]
        pred = pred.reshape(-1,1)
        y_temp = y_temp.reshape(-1,1)
        
        results['MAPE_TS'].append(mape(y_temp[y_temp!=0],pred[y_temp!=0]))
        results['MAE_TS'].append(mae(y_temp,pred))
        results['RMSE_TS'].append(np.sqrt(np.mean((pred-y_temp)**2)))
        
        elapsed_time = time.time() - start_time
        results['TIME'].append(elapsed_time)
        results['I'].append(j)
        print(results,flush = True)
    
    return results



# Evaluation Metrics

def monotonicity(HI,units):
    results = []
    for j in np.unique(units):
        idx = np.ravel(units==j)
        HI_unit = HI[idx]
        dfHI = HI_unit[1:]-HI_unit[:-1]
        dfHI_pos = dfHI[dfHI>0].shape[0]
        dfHI_neg = dfHI[dfHI<=0].shape[0]
        results.append(np.abs(dfHI_neg-dfHI_pos)/len(dfHI-1))
    return np.array(results).mean()




def trend_spearman(HI,T,units):
    results = []
    for j in np.unique(units):
        idx = np.ravel(units==j)
        HI_unit = HI[idx]
        T_unit = T[idx].astype('float64')
        corr = spearmanr(HI_unit,T_unit)[0]
        if np.isnan(corr):
            results.append(0)
        else:
            results.append(corr)
    return np.abs(np.mean(results))


def failure_consistency(HI,units):
    diff = []
    Hi_ends = []
    for j in np.unique(units):
        idx = np.ravel(units==j)
        HI_unit = HI[idx]
        HI_0 = np.mean(HI_unit[0])
        HI_end= np.mean(HI_unit[-1])
        Hi_ends.append(HI_end)
        diff.append(HI_0-HI_end)


    return np.exp(-np.std(Hi_ends)/np.mean(diff))

def mutual_information(HI,RUL,units):
    results = []
    for j in np.unique(units):
        idx = np.ravel(units==j)
        HI_unit = HI[idx]
        RUL_unit = RUL[idx].ravel()
        Mutual_inf = feature_selection.mutual_info_regression(HI_unit,RUL_unit)[0]
        results.append(1-np.exp(-Mutual_inf))
    return np.mean(results)


def mape(real,predicted):
    return np.mean(np.abs((real - predicted) / real)) * 100

def mae(real,predicted):
    return np.mean(np.abs(real - predicted))

def rmse(real,predicted):
    return np.sqrt(np.mean((real - predicted)**2))
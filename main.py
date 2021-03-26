#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from MDN_class import *

def MAPE_calc(percentage_error, error, threshold):
    MPAE_no_threshold = np.mean(percentage_error)
    MAPE = 0
    c = 0
    for i in range(len(error)):
        if error[i] < threshold[i]:
            MAPE += percentage_error[i]
            c    += 1
    if c ==0: 
        MAPE = -1
        return MAPE, MPAE_no_threshold
    else:
        return MAPE/c, MPAE_no_threshold

def visualization(y_test, y_pred_mean, y_pred_std, name):

    counter = np.arange(len(y_test))
    plt.figure(figsize=(20,15))
    plt.plot(counter, y_test, label='Original')
    plt.plot(counter, y_pred_mean,'--', label='Estimated')
    plt.title(name, fontsize=30)
    plt.xlabel('samples', fontsize=20)
    plt.ylabel('output', fontsize=20)
    plt.fill_between(counter, y_pred_mean - y_pred_std, y_pred_mean + y_pred_std, color='b', alpha=0.2)
    plt.legend(fontsize=20)
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    plt.show()

    error = abs(y_test - y_pred_mean.reshape(-1,1))
    Percentage_error = 100*abs(y_test - y_pred_mean.reshape(-1,1))/abs(y_test)

    # plt.figure(figsize=(20,15))
    # plt.plot(counter, error,label='Absolute Error')
    # plt.plot(1.5*y_pred_std, label = '1.5 * $\sigma(x)^2$')
    # plt.legend(fontsize=20)
    # plt.title(name, fontsize=30)
    # plt.xlabel('samples', fontsize=20)
    # plt.ylabel('Absolute error', fontsize=20)
    # plt.xticks(fontsize=30)
    # plt.yticks(fontsize=30)
    # plt.show()

    plt.figure(figsize=(20,15))
    plt.plot(counter, Percentage_error,label='Absolute Percentage Error')
    plt.legend(fontsize=20)
    plt.xlabel('samples', fontsize=20)
    plt.ylabel('% error', fontsize=20)
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    plt.title(name, fontsize=30)
    plt.show()

    threshold = 1.5*y_pred_std
    MAPE_with_threshold, MAPE_without_threshold = MAPE_calc(Percentage_error, error, threshold)
    if type(MAPE_with_threshold) != float:
        return MAPE_with_threshold[0], MAPE_without_threshold
    else:
        return MAPE_with_threshold, MAPE_without_threshold


        




class compressor_scenario_1():
    '''
    Scenario1: 
    In this cell we trian a mixture density networks on each portion of the data
    :: In each portion 75% of random samples are chosen for training and the rest is chosen as the test for accuracy validation

    Soft Sensing scenario 1:
    Predict the Compressor KPIs T_discharge and Motor Current by type one feature engineering which results in the following input space
    ['c_suction_p', 'c_suction_t', 'c_norc_f', 'c_molculard_weight','c_discharge_p', 'Qv_ACFM', 'vapor_flow',]
    '''
    def __init__(self, **kwargs):
        self.portions = kwargs['portions']
        self.input_labels = kwargs['input_labels']
        self.output_labels = kwargs['output_labels']
        self.test_proportion = kwargs['test_proportion']
        self.Execute()

    def Execute(self):
        portion_names = list(self.portions.keys())
        self.MAPE_dic = {}
        for name in portion_names:
            dummy_df = portions[name]
            scaler_dic = {}
            for _output in self.output_labels:
                scanario_name = 'scenario 1_portion ' + name + ' KPI => ' + _output
                X = dummy_df[self.input_labels].values
                Y = dummy_df[_output].values.reshape(-1,1)
                x_train, x_val, y_train, y_val = train_test_split(X[0:int(self.test_proportion*len(X)),:], Y[0:int(self.test_proportion*len(X)),:], train_size=0.85, random_state=42)

                #===== Normalization for each portion individually (This can be gloal to the whole data too, try it later)
                dummy_scaler_x = MinMaxScaler()
                dummy_scaler_y = MinMaxScaler()
                dummy_scaler_x.fit(x_train)
                dummy_scaler_y.fit(y_train)
                scaler_dic[scanario_name] = [dummy_scaler_x, dummy_scaler_y]  ## save the scaler structure

                x_train = dummy_scaler_x.transform(x_train)
                x_val = dummy_scaler_x.transform(x_val)
                y_train = dummy_scaler_y.transform(y_train)
                y_val = dummy_scaler_y.transform(y_val)

                #===== Compiling a mixture density netwrok for each portion seperatly
                sample_MDN = MDN_prediction(no_parameters = 3, components = 10, neurons= 256)
                sample_MDN.fit(x_train = x_train,
                                    y_train = y_train,
                                    x_test = x_val,
                                    y_test = y_val,
                                    epochs = 200,
                                    batch_size = 128)

                x_test = X[int(self.test_proportion*len(X)):,:]
                x_test = dummy_scaler_x.transform(x_test)

                y_test = Y[int(self.test_proportion*len(X)):,:]
                y_test = dummy_scaler_y.transform(y_test)
                y_pred_mean, y_pred_std = sample_MDN.predict_MDN(x_test = x_test)
                MDN_dic[scanario_name] = sample_MDN

                MAPE, MAPE_threshold = visualization(y_test, y_pred_mean, y_pred_std, scanario_name)
                self.MAPE_dic[scanario_name] = [MAPE, MAPE_threshold]

class compressor_scenario_2():
    '''
    Scenario2: 
    Predict the Compressor KPIs T_discharge and Motor Current by type two feature engineering which results in the following input space
    With incorprating estimated H and n as extra engineered features
    sub_Training1: In this part we train a model to predict the H and n
    
    sub_Training2: In this part we use the predicted H and ninside the training and then 
    ['c_suction_p', 'c_suction_t', 'c_norc_f', 'c_molculard_weight','c_discharge_p', 'Qv_ACFM', 'vapor_flow',H, n]
    =============== Assumptions ==============
    *) inside the provided data frames all the actual values for the the inputes and outputes must exist
    *) input_labels_submode1 is those inputes that we would like to use for estimating the intermediate states in the test phase
    *) input_labels_submode2 is those inputes we would like to use for estimating the final output KPIS
    *) output_labelssubmode1 includes all the intermediate states
    *) output_labelssubmode2 includes the main KPIs
    '''
    def __init__(self, **kwargs):
        self.prtions = kwargs['portions']
        self.portion_names = list(portions.keys())
        self.input_labels_submode1 = kwargs['input_labels_submode1']
        self.output_labelssubmode1 = kwargs['output_labelssubmode1']
        self.input_labels_submode2 = kwargs['input_labels_submode2']
        self.output_labelssubmode2 = kwargs['output_labelssubmode2']
        self.test_proportion = kwargs['test_proportion']
        self.submodeling_proportion = kwargs['submodeling_proportion']
        self.Execute_submodeling_1()

    def Execute_submodeling_1(self):  
        self.MAPE_dic = {}
        for name in self.portion_names: ## First we look at each portions
            dummy_df = portions[name]
            scaler_dic = {}
            estimated_states = []
            inermediate_MDN = {} ## This dictionaru is used to save the intermediate states MDN models
            for _output in self.output_labelssubmode1:
                scanario_name = 'scenario2_portion ' + name + ' submodel1 ' +  _output
                X = dummy_df[self.input_labels_submode1].values
                Y = dummy_df[_output].values.reshape(-1,1)

                X_submodel = X[0:int(self.submodeling_proportion*self.test_proportion*len(X)),:]
                Y_submodel = Y[0:int(self.submodeling_proportion*self.test_proportion*len(X)),:]
                x_train, x_val, y_train, y_val = train_test_split(X_submodel, Y_submodel, train_size=0.9, random_state=42)

                #===== Normalization for each portion individually (This can be gloal to the whole data too, try it later)
                dummy_scaler_x = MinMaxScaler()
                dummy_scaler_y = MinMaxScaler()
                dummy_scaler_x.fit(x_train)
                dummy_scaler_y.fit(y_train)
                scaler_dic[scanario_name] = [dummy_scaler_x, dummy_scaler_y]  ## save the scaler structure

                x_train = dummy_scaler_x.transform(x_train)
                x_val = dummy_scaler_x.transform(x_val)
                y_train = dummy_scaler_y.transform(y_train)
                y_val = dummy_scaler_y.transform(y_val)

                #===== Compiling a MDN for each intermediate state seperatly
                sample_MDN = MDN_prediction(no_parameters = 3, components = 10, neurons= 256)
                sample_MDN.fit(x_train = x_train,
                                    y_train = y_train,
                                    x_test = x_val,
                                    y_test = y_val,
                                    epochs = 200,
                                    batch_size = 128)
                inermediate_MDN[scanario_name] = sample_MDN

                x_test = X[int(self.submodeling_proportion*self.test_proportion*len(X)):int(1*self.test_proportion*len(X)),:]
                x_test = dummy_scaler_x.transform(x_test)

                y_test = Y[int(self.submodeling_proportion*self.test_proportion*len(X)):int(1*self.test_proportion*len(X)),:]
                y_test = dummy_scaler_y.transform(y_test)
                y_pred_mean, y_pred_std = sample_MDN.predict_MDN(x_test = x_test)

                counter = np.arange(len(y_test))
                plt.figure(figsize=(20,15))
                plt.plot(counter, y_test, label='Original')
                plt.plot(counter, y_pred_mean,'--', label='Estimated')
                plt.title( scanario_name, fontsize=20)
                plt.xlabel('samples', fontsize=20)
                plt.ylabel('output', fontsize=20)
                plt.fill_between(counter, y_pred_mean - y_pred_std, y_pred_mean + y_pred_std, color='b', alpha=0.2)
                plt.legend(fontsize=20)
                plt.xticks(fontsize=30)
                plt.yticks(fontsize=30)
                plt.show()
                
                dummy_inputes = dummy_scaler_x.transform(X)
                _, dd = sample_MDN.predict_MDN(x_test = dummy_inputes)
                estimated_states.append([dd])
            
            for _kpi in self.output_labelssubmode2:
                scenario_name_main = 'scenario2_portion ' + name + ' submodel2 ' +  _kpi
                final_MDN = {} ## ALL the trained model after incorporating the intermediate states are saved here
                X = dummy_df[self.input_labels_submode2].values ## This is directly comming from the available dataset
                Y = dummy_df[_kpi].values.reshape(-1,1)

                ##===== We use the trained models to estimate the inoutes of this step for H and n
                engineered_features = (np.array(estimated_states[0]).reshape(-1,1), np.array(estimated_states[1]).reshape(-1,1))
                engineered_features = np.hstack(engineered_features)
        
                X_submodel = np.concatenate((X[0:int(self.test_proportion*len(X)),:], engineered_features[0:int(self.test_proportion*len(X)),:]), axis=1)
                Y_submodel = Y[0:int(self.test_proportion*len(X)),:]
                x_train, x_val, y_train, y_val = train_test_split(X_submodel, Y_submodel, train_size=0.9, random_state=42)

                #===== Normalization for each portion individually (This can be gloal to the whole data too, try it later)
                dummy_scaler_x = MinMaxScaler()
                dummy_scaler_y = MinMaxScaler()
                dummy_scaler_x.fit(x_train)
                dummy_scaler_y.fit(y_train)

                x_train = dummy_scaler_x.transform(x_train)
                x_val = dummy_scaler_x.transform(x_val)
                y_train = dummy_scaler_y.transform(y_train)
                y_val = dummy_scaler_y.transform(y_val)

                #===== Compiling a MDN for each intermediate state seperatly
                sample_MDN = MDN_prediction(no_parameters = 3, components = 10, neurons= 256)
                sample_MDN.fit(x_train = x_train,
                                    y_train = y_train,
                                    x_test = x_val,
                                    y_test = y_val,
                                    epochs = 200,
                                    batch_size = 128)
                inermediate_MDN[scanario_name] = sample_MDN

                x_test = np.concatenate((X[int(self.test_proportion*len(X)):,:], engineered_features[int(self.test_proportion*len(X)):,:]), axis=1)
                x_test = dummy_scaler_x.transform(x_test)

                y_test = Y[int(self.test_proportion*len(X)):,:]
                y_test = dummy_scaler_y.transform(y_test)
                y_pred_mean, y_pred_std = sample_MDN.predict_MDN(x_test = x_test)

                MAPE, MAPE_threshold = visualization(y_test, y_pred_mean, y_pred_std, scanario_name)
                self.MAPE_dic[scanario_name] = [MAPE, MAPE_threshold]



#%%

if __name__ == "__main__":

    path = 'C:/Users\Bahador\Desktop\GoogleDrive\Post Doc Works\Honeywell CRD\TeamCodes\CRD-Project-Honeywell\Dummy_Data\Cleaned_data.csv'
    df = pd.read_csv(path)

    num_portions = 4 #Divide the compressor data into 4 different portions
    N = df.shape[0]
    portion_len = N//num_portions
    portions = {}
    for i in range(num_portions):
        portions[str(i)] = pd.DataFrame(df.iloc[i*portion_len:(i+1)*portion_len,:], columns=df.columns, index=df.index[i*portion_len:(i+1)*portion_len])

    scaler_dic = {}
    MDN_dic = {}

    '''
    Scenario1: 
    In this cell we trian a mixture density networks on each portion of the data
    :: In each portion 75% of random samples are chosen for training and the rest is chosen as the test for accuracy validation

    Soft Sensing scenario 1:
    Predict the Compressor KPIs T_discharge and Motor Current by type one feature engineering which results in the following input space
    ['c_suction_p', 'c_suction_t', 'c_norc_f', 'c_molculard_weight','c_discharge_p', 'Qv_ACFM', 'vapor_flow',]
    '''
    input_labels = ['c_suction_p', 'c_suction_t', 'c_norc_f', 'c_molculard_weight','c_discharge_p', 'Qv_ACFM', 'vapor_flow',]
    output_labels = ['m_current', 'c_discharge_t']
    model_scenario1_current = compressor_scenario_1(portions=portions, input_labels=input_labels, output_labels=output_labels, test_proportion=0.8)

    MAPE = model_scenario1_current.MAPE_dic
    MPAE_dict_df = pd.DataFrame(MAPE, index=['MAPE with threshold', 'MAPE without threshold'])
    MPAE_dict_df.plot.bar(rot=0, fontsize=20, figsize=(20,15))



    '''
    Scenario2: 
    Predict the Compressor KPIs T_discharge and Motor Current by type two feature engineering which results in the following input space
    With incorprating estimated H and n as extra engineered features
    sub_Training1: In this part we train a model to predict the H and n
    
    sub_Training2: In this part we use the predicted H and ninside the training and then 
    ['c_suction_p', 'c_suction_t', 'c_norc_f', 'c_molculard_weight','c_discharge_p', 'Qv_ACFM', 'vapor_flow',H, n]
    '''
    input_labels_submode1 = ['c_suction_p','c_discharge_p', 'c_suction_t', 'c_norc_f', 'c_molculard_weight']
    output_labelssubmode1 = ['n', 'Operating_head']
    input_labels_submode2 = ['c_suction_p', 'c_suction_t', 'c_norc_f', 'c_molculard_weight','c_discharge_p', 'Qv_ACFM', 'vapor_flow']
    output_labelssubmode2 = ['c_discharge_t', 'm_current']

    model_scenario2_H_va_n = compressor_scenario_2(portions=portions, input_labels_submode1 = input_labels_submode1,
                                output_labelssubmode1 = output_labelssubmode1,
                                input_labels_submode2 = input_labels_submode2,
                                output_labelssubmode2 = output_labelssubmode2,
                                test_proportion = 0.8,
                                submodeling_proportion = 0.7)
    
    MAPE = model_scenario2_H_va_n.MAPE_dic
    MPAE_dict_df = pd.DataFrame(MAPE, index=['MAPE with threshold', 'MAPE without threshold'])
    MPAE_dict_df.plot.bar(rot=0, fontsize=20, figsize=(20,15))


#%%





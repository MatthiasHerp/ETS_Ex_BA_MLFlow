import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import os
import sys
import math
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.optimize import minimize 
import mlflow
import mlflow.pyfunc
import cloudpickle
import traceback


#defining all required functions

def days_around_events(exogen, before, after):
    df_before = pd.DataFrame()
    df_before['date'] = exogen.index

    for event in exogen.columns:
        # saving the column number of the event
        event_col_number = exogen.columns.get_loc(event)
        # defining the number of days before as an array depending on the number on the before array definde by the user
        # Example: event: SNAP_CA --> col_number=0, first position in before=2 (1 & 2 days before), before_event=[1,2]
        before_event = [i for i in range(1, before[event_col_number] + 1)]
        for d in before_event:
            day_before = list()
            for i in range(0, len(exogen.index) - d):
                if exogen.iloc[i + d, exogen.columns.get_loc(event)] == 1:
                    day_before.append(1)
                else:
                    day_before.append(0)

            day_before.append(np.zeros(d))
            day_before = np.concatenate(day_before, axis=None)
            df_before[str(str(d) + '_days_before_' + event)] = day_before

    # making the date the index for better merging
    df_before.index = df_before['date']
    # deleting the data column as we dont want it as explanatory variable
    del df_before['date']

    df_after = pd.DataFrame()
    df_after['date'] = exogen.index

    for event in exogen.columns:
        # saving the column number of the event
        event_col_number = exogen.columns.get_loc(event)
        after_event = [i for i in range(1, after[event_col_number] + 1)]
        for d in after_event:
            day_after = list()
            day_after.append(np.zeros(d))
            for i in range(d, len(exogen.index)):
                if exogen.iloc[i - d, exogen.columns.get_loc(event)] == 1:
                    day_after.append(1)
                else:
                    day_after.append(0)

            day_after = np.concatenate(day_after, axis=None)
            df_after[str(str(d) + '_days_after_' + event)] = day_after
    # making the date the index for better merging
    df_after.index = df_after['date']
    # deleting the data column as we dont want it as explanatory variable
    del df_after['date']

    # merging the exogen variable dataframe with the day before data frame
    # Note: both merging at the end to aviod problems such as days after days before....
    exogen = pd.merge(exogen, df_before, left_index=True, right_index=True)
    exogen = pd.merge(exogen, df_after, left_index=True, right_index=True)

    return exogen

# Defining the Model
def model(params, y, exog):
    alpha = params[0]
    beta = params[1]
    gamma = params[2]
    omega = params[3]
    l_init_HM = params[4]
    b_init_HM = params[5]
    s_init_HM = np.vstack(params[6:13])
    reg = (params[13:13 + len(exogen.columns)])

    # Note: added len(exog) as now we have variable number of exog variables due to days before and after
    #      before: params[13:18] as we have 5 types of events

    print('alpha:', alpha, 'beta:', beta, 'gamma:', gamma, 'omega:', omega,
          l_init_HM, b_init_HM, s_init_HM,
          'reg:', reg)

    results = ETS_M_Ad_M(alpha, beta, gamma, omega,
                         l_init_HM, b_init_HM, s_init_HM, reg, y, exog)

    error_list = results['errors_list']
    error_list = [number ** 2 for number in error_list]
    error_sum = sum(error_list)
    print(error_sum)

    return error_sum

    # Defining the model step: calculate new estimates


def calc_new_estimates(l_past, b_past, s_past, alpha, beta, omega, gamma, e, weekly_transition_matrix,
                       weekly_update_vector):
    try:
        l = (l_past + omega * b_past) * (1 + alpha * e)
        b = omega * b_past + beta * (l_past + omega * b_past) * e
        s = np.dot(weekly_transition_matrix, s_past) + weekly_update_vector * gamma * e
    except:
        print('lpast: ', l_past)
        print('bpast', b_past)
        print('spast', s_past)
        print('alpha', alpha)
        print('beta', beta)
        print('omeag', omega)
        print('gamma', gamma)
        print('error', e)

    return l, b, s


# Define the model step: calculate errors
def calc_error(l_past, b_past, s_past, omega, y, i, reg, exog):
    mu = (l_past + omega * b_past) * s_past[0] + np.dot(reg, exog.iloc[i]) * (l_past + omega * b_past) * s_past[0]

    e = (y[i] - mu) / y[i]

    e_absolute = y[i] - mu

    return mu, e, e_absolute


# Define the model step: save estimates
def save_estimates(errors_list, point_forecast, l_list, b_list, s_list, e_absolute, mu, l_past, b_past, s_past):
    errors_list.append(e_absolute)
    point_forecast.append(mu)
    l_list.append(l_past)
    b_list.append(b_past)
    s_list.append(s_past[0])

    return errors_list, point_forecast, l_list, b_list, s_list


# Define the required transitional matrices and vectors of the model
def seasonal_matrices():
    col_1 = np.vstack(np.zeros(6))
    col_2_6 = np.identity(6)
    matrix_6 = np.hstack((col_1, col_2_6))
    row_7 = np.concatenate((1, np.zeros(6)), axis=None)
    weekly_transition_matrix = np.vstack((matrix_6, row_7))

    weekly_update_vector = np.vstack(np.concatenate((np.zeros(6), 1), axis=None))

    return weekly_transition_matrix, weekly_update_vector


# Defining the fit calculator of the model comnbining the above sub functions and being called by model
def ETS_M_Ad_M(alpha, beta, gamma, omega,
               l_init_HM, b_init_HM, s_init_HM, reg, y, exog):
    t = len(y)
    errors_list = list()
    point_forecast = list()
    l_list = list()
    b_list = list()
    s_list = list()

    # Initilaisation
    l_past = l_init_HM
    b_past = b_init_HM
    s_past = s_init_HM

    # defining the seasonal matrices for the calculation of new state estimates
    weekly_transition_matrix, weekly_update_vector = seasonal_matrices()

    # computation loop:
    for i in range(0, t):
        # compute one step ahead  forecast for timepoint i
        mu, e, e_absolute = calc_error(l_past, b_past, s_past, omega, y, i, reg, exog)

        # save estimation error for Likelihood computation as well as the states and forecasts (fit values)
        errors_list, point_forecast, l_list, b_list, s_list = save_estimates(errors_list, point_forecast, l_list,
                                                                             b_list, s_list,
                                                                             e_absolute, mu, l_past, b_past, s_past)

        # Updating all state estimates with the information set up to time point i
        l, b, s = calc_new_estimates(l_past, b_past, s_past, alpha, beta, omega, gamma, e, weekly_transition_matrix,
                                     weekly_update_vector)

        # denote updated states from i as past states for time point i+1 in the next iteration of the loop
        l_past = l
        b_past = b
        s_past = s

    return {'errors_list': errors_list, 'point forecast': point_forecast,
            'l_list': l_list, 'b_list': b_list, 's_list': s_list}


# Defining a function that returns fit values for estiamted parameters
def fit_extracter(params, y, exog):
    alpha = params[0]
    beta = params[1]
    gamma = params[2]
    omega = params[3]
    l_init_HM = params[4]
    b_init_HM = params[5]
    s_init_HM = np.vstack(params[6:13])
    reg = (params[13:13 + len(exogen.columns)])

    # Note: added len(exog) as now we have variable number of exog variables due to days before and after
    #      before: params[13:18] as we have 5 types of events

    results = ETS_M_Ad_M(alpha, beta, gamma, omega,
                         l_init_HM, b_init_HM, s_init_HM, reg, y, exog)

    return results


# Defining a function to extracte forecasts
def forecasting(params, exog, h):
    alpha = params[0]
    beta = params[1]
    gamma = params[2]
    omega = params[3]
    l_init_HM = params[4]
    b_init_HM = params[5]
    s_init_HM = np.vstack(params[6:13])
    reg = (params[13:13 + len(exogen.columns)])

    # Note: added len(exog) as now we have variable number of exog variables due to days before and after
    #      before: params[13:18] as we have 5 types of events

    results = ETS_M_Ad_M_forecast(alpha, beta, gamma, omega,
                                  l_init_HM, b_init_HM, s_init_HM, reg, h, exog)

    return results


# defining a function computing point forecasts
def ETS_M_Ad_M_forecast(alpha, beta, gamma, omega,
                        l_init_HM, b_init_HM, s_init_HM, reg, h, exog):
    # computing the number of time points as the length of the forecasting vector
    t = h
    point_forecast = list()
    l_list = list()
    b_list = list()
    s_list = list()

    # Initilaisation
    l_past = l_init_HM
    b_past = b_init_HM
    s_past = s_init_HM

    # defining the seasonal matrices for the calculation of new state estimates
    weekly_transition_matrix, weekly_update_vector = seasonal_matrices()

    # computation loop:
    for i in range(1, h + 1):
        # compute one step ahead  forecast for timepoint t
        mu = (l_past + omega * b_past) * s_past[0] + np.dot(reg, exog.iloc[i - 1]) * (l_past + omega * b_past) * s_past[
            0]

        point_forecast.append(mu)
        l_list.append(l_past)
        b_list.append(b_past)
        s_list.append(s_past[0])

        s_past = np.dot(weekly_transition_matrix, s_past)

    return {'point forecast': point_forecast,
            'l_list': l_list, 'b_list': b_list, 's_list': s_list}




if __name__ == "__main__":
   
   
   #Importing the given before and after parameters
   before = [int(i) for i in sys.argv[1:6]]
   after = [int(i) for i in sys.argv[6:11]]
   sys_string = str(sys.argv)
   
   #reading in the revenue_CA_1_FOODS_day time series csv
   revenue_CA_1_FOODS_day = os.path.join(os.path.dirname(os.path.abspath(__file__)), "revenue_CA_1_FOODS_day.csv")
   revenue_CA_1_FOODS_day = pd.read_csv(revenue_CA_1_FOODS_day, index_col='date')

   #defining the training and evaluation set
   y = revenue_CA_1_FOODS_day[:-365]
   y_predict = revenue_CA_1_FOODS_day[-365:]

   #reading in the exogen variables which are the SNAP, Sporting, Cultural, National and Religious events
   exogen = os.path.join(os.path.dirname(os.path.abspath(__file__)), "exogen_variables.csv")
   exogen = pd.read_csv(exogen, index_col='date')
    
   # Include days before and after events into the exogen data set
   exogen = days_around_events(exogen, before, after)

   # Define training and prediction data sets for the exogen variables
   exog_to_train = exogen.iloc[:(len(revenue_CA_1_FOODS_day) - 365)]
   exog_to_test = exogen.iloc[(len(revenue_CA_1_FOODS_day) - 365):]

    #setting the experiment
    #mlflow.set_experiment("ETS_Exog_B_A")
    
   # Useful for multiple runs (only doing one run in this sample notebook)    
   with mlflow.start_run() as run: #as run so that i can get the id later on for logging the model
        #Defining Starting Parameters
        #Optimal Starting parameters after running the starting parameters calculated by the Hyndman method for two iterations
        
        Starting_Parameters_optimal = [ 2.32625532e-01,  1.00000000e-06,  1.41907946e-02,  9.99333847e-01,
                5.55499458e+03,  3.96440052e+01,  1.14589164e+00,  1.18053933e+00,
                8.78903981e-01,  7.82677252e-01,  7.54118200e-01,  7.76617802e-01,
                9.27728973e-01,  1.28533624e-01, -5.34822743e-02, -1.50822221e-01,
                1.44746722e-02,  1.25113251e-02, np.zeros(len(exogen.columns)-5)]
        
        Starting_Parameters_optimal = np.concatenate(Starting_Parameters_optimal,axis=None)
  

        #Note: adding zeros for days before and after by the length of exogen. -5 because we have starting values for the event days
        #      concatenate because array in array

        

        #Defining bounds
        bounds = [(0.000001,0.9999),(0.000001,0.9999),(0.000001,0.9999),(0.000001,0.9999),(-np.inf,np.inf),(-np.inf,np.inf),
                                (-2,2),(-2,2),(-2,2),(-2,2),(-2,2),(-2,2),(-2,2),
                  (-1,np.inf),(-1,np.inf),(-1,np.inf),(-1,np.inf),(-1,np.inf)]
        
        #adding bounds for the increased number of exogen
        #adding one bound for each additional day before and after
        for i in range(0,sum(before)+sum(after)):
            bounds.append((-1,np.inf))
        
        #Saving Parameters

        #mlflow.log_param("before", before)

        #mlflow.log_param("after", after)

        #mlflow.log_param("sys_string", sys_string)

        #mlflow.log_param("exog", exogen.iloc[1])
        
        #mlflow.log_param("bounds", bounds)
        
        mlflow.log_param("Starting_Parameters_optimal", Starting_Parameters_optimal)

        #running the model optimization
        res = minimize(model, Starting_Parameters_optimal, args=(np.array(y['revenue']), exog_to_train), 
                       method='L-BFGS-B', bounds = bounds)
        
        #logging in optimal parameters
        mlflow.log_param("Model_Parameters_optimal", res.x)

        #the fit extracter is run with the optimal values optained from the optimizer (res.x) and the time series y
        fit = fit_extracter(res.x, np.array(y['revenue']), exog_to_train)

        #creating a data frame with the time series as date object and index
        fit_values = pd.DataFrame({'fitted' : fit['point forecast'], 'date' : pd.to_datetime(y.index)})
        fit_values = fit_values.set_index('date')

        #Plotting results
        revenue_CA_1_FOODS_day.index =pd.to_datetime(revenue_CA_1_FOODS_day.index)
        #Plot the fit and the training data set
        plt.figure(figsize=(15, 5))
        plt.plot(revenue_CA_1_FOODS_day[:-365], color = 'blue')
        plt.plot(fit_values, color="green")
        plt.xlabel("date")
        plt.ylabel("revenue_CA_1_FOODS")
        plt.legend(("realization", "fitted"), loc="upper left")
        plt.savefig('fit_total_plot.png')
       
        mlflow.log_artifact("./fit_total_plot.png", "plots") #adds it to the plot folder

        #Plot the fitted and training data set fpr the first year
        plt.figure(figsize=(15, 5))
        plt.plot(revenue_CA_1_FOODS_day[:366])
        plt.plot(fit_values[:366], color="green")
        plt.xlabel("date")
        plt.ylabel("revenue_CA_1_FOODS")
        plt.legend(("realization", "fitted"), loc="upper left")
        plt.savefig('fit_1year_plot.png')
        
        mlflow.log_artifact("./fit_1year_plot.png", "plots")

        #extracting the last (most recent) values of the states for forecasting
        l_values = fit['l_list'][len(fit['l_list'])-1:]
        b_values = fit['b_list'][len(fit['b_list'])-1:]
        s_values = fit['s_list'][len(fit['s_list'])-7:]

        #creating a list of all optimal parameters for forecasting
        forecast_parameters = np.concatenate([res.x[0:4],l_values,b_values,s_values,res.x[13:13+len(exogen.columns)]],
                                             axis=None)
        #logging the forecasting parameters
        mlflow.log_param("Model_Forecasting_Parameters_optimal", forecast_parameters)


        #Note: added len(exog) as now we have variable number of exog variables due to days before and after
        #      before: params[13:18] as we have 5 types of events
        
        #computing forecasts for the horizons 7,14,21,31,365 and saving the results
        forecasts_7 = forecasting(forecast_parameters, exog_to_test, 7)
        forecasts_14 = forecasting(forecast_parameters, exog_to_test, 14)
        forecasts_21 = forecasting(forecast_parameters, exog_to_test, 21)
        forecasts_31 = forecasting(forecast_parameters, exog_to_test, 31)
        forecasts_365 = forecasting(forecast_parameters, exog_to_test, 365)

        #creating a data frame with the time series as date object and index for each forecasting horizon
        #Note: np.concatenate the data to form it into one array

        forecasted_values_7 = pd.DataFrame({'forecast' : np.concatenate(forecasts_7['point forecast'],axis=None),
                                              'date' : pd.to_datetime(y_predict.index[:7])})
        forecasted_values_7 = forecasted_values_7.set_index('date')

        forecasted_values_14 = pd.DataFrame({'forecast' : np.concatenate(forecasts_14['point forecast'],axis=None),
                                              'date' : pd.to_datetime(y_predict.index[:14])})
        forecasted_values_14 = forecasted_values_14.set_index('date')

        forecasted_values_21 = pd.DataFrame({'forecast' : np.concatenate(forecasts_21['point forecast'],axis=None),
                                              'date' : pd.to_datetime(y_predict.index[:21])})
        forecasted_values_21 = forecasted_values_21.set_index('date')

        forecasted_values_31 = pd.DataFrame({'forecast' : np.concatenate(forecasts_31['point forecast'],axis=None),
                                              'date' : pd.to_datetime(y_predict.index[:31])})
        forecasted_values_31 = forecasted_values_31.set_index('date')

        forecasted_values_365 = pd.DataFrame({'forecast' : np.concatenate(forecasts_365['point forecast'],axis=None),
                                              'date' : pd.to_datetime(y_predict.index)})
        forecasted_values_365 = forecasted_values_365.set_index('date')

        #Plot the results over the entire time span
        plt.figure(figsize=(15, 5))
        plt.plot(revenue_CA_1_FOODS_day, color = 'blue')
        plt.plot(fit_values, color="green")
        plt.plot(forecasted_values_365, color="red")
        plt.xlabel("date")
        plt.ylabel("revenue_CA_1_FOODS")
        plt.legend(("realization", "fitted","forecast"), loc="upper left")
        plt.savefig('fit_forecast_total_plot.png')
        
        mlflow.log_artifact("./fit_forecast_total_plot.png", "plots")

        #make sure the prediction data set index is a date variable for plotting
        y_predict.index =pd.to_datetime(y_predict.index)

        #Plot the first 31 days of the prediction data and their forecasts
        plt.figure(figsize=(15, 5))
        plt.plot(y_predict[:31])
        plt.plot(forecasted_values_31, color="red")
        plt.xlabel("date")
        plt.ylabel("revenue_CA_1_FOODS")
        plt.legend(("realization", "forecast"), loc="upper left")
        plt.savefig('fit_forecast_31days_plot.png')
        
        mlflow.log_artifact("./fit_forecast_31days_plot.png", "plots")
        
        y_prediction_horizons = (y_predict['revenue'][:7],y_predict['revenue'][:14],
                                      y_predict['revenue'][:21],y_predict['revenue'][:31],
                                      y_predict['revenue'])
        
        forecast_values_horizons = (forecasted_values_7['forecast'], forecasted_values_14['forecast'],
                                        forecasted_values_21['forecast'], forecasted_values_31['forecast'],
                                        forecasted_values_365['forecast'])
        horizon = (7,14,21,31,365)
        
        for i in range(0,5):
            mlflow.log_metric(key = "RMSE", 
                              value = np.sqrt(mean_squared_error(y_prediction_horizons[i], forecast_values_horizons[i])), 
                              step = horizon[i])
        
        for i in range(0,5):
            mlflow.log_metric(key = "R2", 
                              value = r2_score(y_prediction_horizons[i], forecast_values_horizons[i]), 
                              step = horizon[i])
            
        for i in range(0,5):
            mlflow.log_metric(key = "MAE", 
                              value = mean_absolute_error(y_prediction_horizons[i], forecast_values_horizons[i]), 
                              step = horizon[i])
        
        #Saving Parameters
        #mlflow.log_param("before", before)
        #mlflow.log_param("after", after)
        #mlflow.log_param("sys_string", sys_string)
        #mlflow.log_param("exog", exogen.iloc[1])
                            
        #Saving optimal Parameters as csv artifact
        #Optimum_Parameters = pd.DataFrame(res.x)
        #Optimum_Parameters.to_csv('Optimum_Parameters.csv') 
        #mlflow.log_artifact("./Optimum_Parameters.csv")
        
        # Define the model class
        class ETS_Exogen(mlflow.pyfunc.PythonModel):
            #constructor defines attributes of the class
            #here we have a model thus we have attributes being the parameters of the model
            #we pass the parameter alpha used in the prediction
            #we pass the forecasting horizon to have a parameter data scientists calling the model would be interested at changing

            def __init__(self, params, before, after):
                self.params = params
                self.before = before
                self.after = after
                #self.h = h

            def load_context(self, context):
                    import numpy as np
                    import pandas as pd #data wrangeling
                    url_to_exogen_raw = 'https://raw.githubusercontent.com/MatthiasHerp/ETS_Ex_BA_MLFlow/master/exogen_variables.csv'
                    self.exogen = pd.read_csv(url_to_exogen_raw, index_col='date')


            def predict(self, context, model_input):

                def seasonal_matrices():

                    #defining weekly transition matrix:
                    #1. defining first column of zeros (1 row to short)
                    col_1 = np.vstack(np.zeros(6))
                    #2. defining identity matrix 1 row and column to small
                    col_2_6 = np.identity(6)
                    #3. adding the 1 column and the identity matrix, now all states are updated to jump up one step in the state vector
                    matrix_6 = np.hstack((col_1,col_2_6))
                    #4. creating a final row in which the current state is put in last place and will be added by an update
                    row_7 = np.concatenate((1,np.zeros(6)), axis = None)
                    #5. adding the last row to the matrix to make it complete
                    weekly_transition_matrix = np.vstack((matrix_6,row_7))

                    #defining the weekly updating vector
                    weekly_update_vector = np.vstack(np.concatenate((np.zeros(6),1), axis = None))

                    return weekly_transition_matrix, weekly_update_vector

                def ETS_M_Ad_M_forecast(alpha,beta,gamma,omega,
                  l_init_HM,b_init_HM,s_init_HM,reg,h,exogen):

                    #computing the number of time points as the length of the forecasting vector
                    t = h
                    point_forecast = list()

                    #Initilaisation
                    l_past = l_init_HM
                    b_past = b_init_HM
                    s_past = s_init_HM

                    #defining the seasonal matrices for the calculation of new state estimates
                    weekly_transition_matrix, weekly_update_vector = seasonal_matrices()


                    #computation loop:
                    for i in range(1,h+1): #+1 because range(1,31): 1,..30, thus range(1,31+1): 1,...,31

                        #compute one step ahead  forecast for timepoint t
                        mu = (l_past + omega * b_past) * s_past[0] + np.dot(reg,exogen.iloc[i-1]) * (l_past + omega * b_past) * s_past[0]

                        point_forecast.append(mu)

                        s_past = np.dot(weekly_transition_matrix,s_past)

                    return  point_forecast

                def days_around_events(exogen, before, after):

                    df_before = pd.DataFrame()
                    df_before['date'] = exogen.index


                    for event in exogen.columns:
                        #saving the column number of the event
                        event_col_number = exogen.columns.get_loc(event)
                        #defining the number of days before as an array depending on the number on the before array definde by the user
                        #Example: event: SNAP_CA --> col_number=0, first position in before=2 (1 & 2 days before), before_event=[1,2]
                        before_event = [i for i in range(1,before[event_col_number]+1)]
                        for d in before_event:
                            day_before = list()
                            for i in range(0,len(exogen.index)-d): 
                                    if exogen.iloc[i+d,exogen.columns.get_loc(event)] == 1:
                                        day_before.append(1)
                                    else:
                                        day_before.append(0)

                            day_before.append(np.zeros(d))
                            day_before=np.concatenate(day_before,axis=None)
                            df_before[str(str(d)+'_days_before_'+event)] = day_before

                    #making the date the index for better merging
                    df_before.index = df_before['date']
                    #deleting the data column as we dont want it as explanatory variable
                    del df_before['date']


                    df_after = pd.DataFrame()
                    df_after['date'] = exogen.index


                    for event in exogen.columns:
                        #saving the column number of the event
                        event_col_number = exogen.columns.get_loc(event)
                        after_event = [i for i in range(1,after[event_col_number]+1)]
                        for d in after_event:
                            day_after = list()
                            day_after.append(np.zeros(d))
                            for i in range(d,len(exogen.index)): 
                                    if exogen.iloc[i-d,exogen.columns.get_loc(event)] == 1:
                                        day_after.append(1)
                                    else:
                                        day_after.append(0)

                            day_after=np.concatenate(day_after,axis=None)
                            df_after[str(str(d)+'_days_after_'+event)] = day_after
                    #making the date the index for better merging
                    df_after.index = df_after['date']
                    #deleting the data column as we dont want it as explanatory variable
                    del df_after['date']

                    #merging the exogen variable dataframe with the day before data frame
                    #Note: both merging at the end to aviod problems such as days after days before....
                    exogen = pd.merge(exogen, df_before, left_index=True, right_index=True)
                    exogen = pd.merge(exogen, df_after, left_index=True, right_index=True)

                    return exogen

                #pass on exogen data
                exogen = days_around_events(self.exogen, self.before, self.after)


                alpha = self.params[0] 
                beta = self.params[1]
                gamma = self.params[2]
                omega = self.params[3]
                l_init_HM = self.params[4]
                b_init_HM = self.params[5]
                s_init_HM = np.vstack(self.params[6:13])
                reg = (self.params[13:13+len(exogen.columns)]) #we need the file exogen columns

                #model input is the forecasting horizon
                h=model_input

                results = ETS_M_Ad_M_forecast(alpha,beta,gamma,omega,
                      l_init_HM,b_init_HM,s_init_HM,reg,h,exogen) #note I changed exog to exogen here as well as in the hwl function

                return np.concatenate(results,axis=None) #i concatenate because otherwise each result is one array
            
        # Create an `artifacts` dictionary that assigns a unique name to the saved XGBoost model file.
        # This dictionary will be passed to `mlflow.pyfunc.save_model`, which will copy the model file
        # into the new MLflow model's directory.

        #artifacts = {
        #    "exogen_variables": "/Users/mah/Desktop/M5_Wallmart_Challenge/exogen_variables.csv"
        #}
        #os.path.join(os.path.dirname(os.path.abspath(__file__)), "exogen_variables.csv")
        #"/Users/mah/Desktop/M5_Wallmart_Challenge/exogen_variables.csv"

        #how do i set the directory so he understands where the file is? because it comes from a repository?

        # Create a Conda environment for the new MLflow model that contains the XGBoost library
        # as a dependency, as well as the required CloudPickle library
        
        conda_env = {
            'channels': ['defaults'],
            'dependencies': [
                'numpy={}'.format(np.__version__),
                'pandas={}'.format(pd.__version__),
                'cloudpickle={}'.format(cloudpickle.__version__)
            ],
            'name': 'naiv_env'
        }

        ETS_Exogen = ETS_Exogen(params=forecast_parameters, before=before,after=after) #taking forecasting parameters from the model & last available day
        #mlflow.pyfunc.log_model(python_model=ETS_Exogen, conda_env=conda_env, artifacts=artifacts)
        #mlflow.pyfunc.save_model(path=model_path, python_model=ETS_Exogen, conda_env=conda_env, artifacts=artifacts)
        # exception handling 
        try:
            #run_id = run.info.run_id
            #mlflow.log_param("run_id", run_id)
            #mlflow.log_param("path", str("runs:/"+run_id+"/artifacts/"))
            mlflow.pyfunc.log_model(artifact_path="model",python_model=ETS_Exogen, conda_env=conda_env)#, artifacts=artifacts)
        except: 
            # save stack trace
            stack_trace = traceback.format_exc()
            mlflow.log_param("stack trace", stack_trace)

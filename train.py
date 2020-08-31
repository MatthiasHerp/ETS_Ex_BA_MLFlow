import numpy as np
import pandas as pd
#import statsmodels.api as sm
import matplotlib
#import matplotlib.pyplot as plt
import os
#import datetime as dt
#import statsmodels as sm
#import math
#from statistics import mean 
#from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
#from scipy.optimize import minimize #LinearConstraint, NonlinearConstraint
import mlflow

'''

if __name__ == "__main__":


   

    #reading in the revenue_CA_1_FOODS_day time series csv
    revenue_CA_1_FOODS_day = os.path.join(os.path.dirname(os.path.abspath(__file__)), "revenue_CA_1_FOODS_day.csv")
    revenue_CA_1_FOODS_day = pd.read_csv(revenue_CA_1_FOODS_day, index_col='date')

    #defining the training and evaluation set
    y = revenue_CA_1_FOODS_day[:-365]
    y_predict = revenue_CA_1_FOODS_day[-365:]

    #reading in the exogen variables which are the SNAP, Sporting, Cultural, National and Religious events
    exogen = os.path.join(os.path.dirname(os.path.abspath(__file__)), "exogen_variables.csv")
    exogen = pd.read_csv(exogen, index_col='date')


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

    #Include days before and after events into the exogen data set
    exogen = days_around_events(exogen, before, after)

    #Define training and prediction data sets for the exogen variables
    exog_to_train = exogen.iloc[:(len(revenue_CA_1_FOODS_day)-365)]
    exog_to_test = exogen.iloc[(len(revenue_CA_1_FOODS_day)-365):]

    #Defining the Model
    def model(params, y, exog):

        alpha = params[0]
        beta = params[1]
        gamma = params[2]
        omega = params[3]
        l_init_HM = params[4]
        b_init_HM = params[5]
        s_init_HM = np.vstack(params[6:13])
        reg = (params[13:13+len(exogen.columns)])
    
        #Note: added len(exog) as now we have variable number of exog variables due to days before and after
        #      before: params[13:18] as we have 5 types of events
        
        
        print('alpha:', alpha,'beta:', beta,'gamma:', gamma, 'omega:', omega,
              l_init_HM,b_init_HM,s_init_HM,
             'reg:',reg)

        results = ETS_M_Ad_M(alpha,beta,gamma,omega,
              l_init_HM,b_init_HM,s_init_HM,reg,y,exog)

        error_list = results['errors_list']
        error_list = [number ** 2 for number in error_list]
        error_sum = sum(error_list)
        print(error_sum)                     

        return error_sum

    
        #Defining the model step: calculate new estimates
    def calc_new_estimates(l_past, b_past, s_past, alpha, beta, omega, gamma, e, weekly_transition_matrix, weekly_update_vector):

        try:
            l = (l_past + omega * b_past) * (1 + alpha * e)
            b = omega * b_past + beta * (l_past + omega * b_past) * e
            s = np.dot(weekly_transition_matrix,s_past) + weekly_update_vector * gamma * e
        except:
            print('lpast: ',l_past) 
            print('bpast', b_past) 
            print('spast', s_past) 
            print('alpha', alpha) 
            print('beta', beta) 
            print('omeag', omega) 
            print('gamma', gamma) 
            print('error', e)

        return l,b,s


    #Define the model step: calculate errors
    def calc_error(l_past, b_past, s_past, omega, y, i, reg, exog):

        mu = (l_past + omega * b_past) * s_past[0] + np.dot(reg,exog.iloc[i]) * (l_past + omega * b_past) * s_past[0]

        e = (y[i] - mu) / y[i]

        e_absolute = y[i] - mu

        return mu, e, e_absolute

    
    #Define the model step: save estimates
    def save_estimates(errors_list,point_forecast,l_list,b_list,s_list,e_absolute,mu,l_past,b_past,s_past):

        errors_list.append(e_absolute) 
        point_forecast.append(mu)
        l_list.append(l_past)
        b_list.append(b_past)
        s_list.append(s_past[0])

        return errors_list,point_forecast,l_list,b_list,s_list


    #Define the required transitional matrices and vectors of the model
    def seasonal_matrices():
        
        col_1 = np.vstack(np.zeros(6))
        col_2_6 = np.identity(6)
        matrix_6 = np.hstack((col_1,col_2_6))
        row_7 = np.concatenate((1,np.zeros(6)), axis = None)
        weekly_transition_matrix = np.vstack((matrix_6,row_7))
        
        weekly_update_vector = np.vstack(np.concatenate((np.zeros(6),1), axis = None))

        return weekly_transition_matrix, weekly_update_vector

    #Defining the fit calculator of the model comnbining the above sub functions and being called by model
    def ETS_M_Ad_M(alpha,beta,gamma,omega,
              l_init_HM,b_init_HM,s_init_HM,reg,y,exog):

        t = len(y)
        errors_list = list()
        point_forecast = list()
        l_list = list()
        b_list = list()
        s_list = list()

        #Initilaisation
        l_past = l_init_HM
        b_past = b_init_HM
        s_past = s_init_HM

        #defining the seasonal matrices for the calculation of new state estimates
        weekly_transition_matrix, weekly_update_vector = seasonal_matrices()


        #computation loop:
        for i in range(0,t): 

            #compute one step ahead  forecast for timepoint i
            mu, e, e_absolute = calc_error(l_past, b_past, s_past, omega, y, i, reg, exog)

            #save estimation error for Likelihood computation as well as the states and forecasts (fit values)
            errors_list,point_forecast,l_list,b_list,s_list = save_estimates(errors_list,point_forecast,l_list,b_list,s_list,
                                                                             e_absolute,mu,l_past,b_past,s_past)


            #Updating all state estimates with the information set up to time point i
            l,b,s = calc_new_estimates(l_past, b_past, s_past, alpha, beta, omega, gamma, e, weekly_transition_matrix, weekly_update_vector)


            #denote updated states from i as past states for time point i+1 in the next iteration of the loop
            l_past = l
            b_past = b
            s_past = s


        return  {'errors_list' : errors_list, 'point forecast' : point_forecast,
                 'l_list' : l_list, 'b_list' : b_list, 's_list' : s_list}
    
    
    #Defining a function that returns fit values for estiamted parameters
    def fit_extracter(params, y, exog):

            alpha = params[0] 
            beta = params[1]
            gamma = params[2]
            omega = params[3]
            l_init_HM = params[4]
            b_init_HM = params[5]
            s_init_HM = np.vstack(params[6:13]) 
            reg = (params[13:13+len(exogen.columns)])
    
            #Note: added len(exog) as now we have variable number of exog variables due to days before and after
            #      before: params[13:18] as we have 5 types of events

            results = ETS_M_Ad_M(alpha,beta,gamma,omega,
                  l_init_HM,b_init_HM,s_init_HM,reg,y,exog)

            return results
    
    
    #Defining a function to extracte forecasts
    def forecasting(params, exog, h):

            alpha = params[0] 
            beta = params[1]
            gamma = params[2]
            omega = params[3]
            l_init_HM = params[4]
            b_init_HM = params[5]
            s_init_HM = np.vstack(params[6:13])
            reg = (params[13:13+len(exogen.columns)])
    
            #Note: added len(exog) as now we have variable number of exog variables due to days before and after
            #      before: params[13:18] as we have 5 types of events

            results = ETS_M_Ad_M_forecast(alpha,beta,gamma,omega,
                  l_init_HM,b_init_HM,s_init_HM,reg,h,exog)

            return results

    
    #defining a function computing point forecasts
    def ETS_M_Ad_M_forecast(alpha,beta,gamma,omega,
                  l_init_HM,b_init_HM,s_init_HM,reg,h,exog):

            #computing the number of time points as the length of the forecasting vector
            t = h
            point_forecast = list()
            l_list = list()
            b_list = list()
            s_list = list()

            #Initilaisation
            l_past = l_init_HM
            b_past = b_init_HM
            s_past = s_init_HM

            #defining the seasonal matrices for the calculation of new state estimates
            weekly_transition_matrix, weekly_update_vector = seasonal_matrices()


            #computation loop:
            for i in range(1,h+1):

                #compute one step ahead  forecast for timepoint t
                mu = (l_past + omega * b_past) * s_past[0] + np.dot(reg,exog.iloc[i-1]) * (l_past + omega * b_past) * s_past[0]
            
                point_forecast.append(mu)
                l_list.append(l_past)
                b_list.append(b_past)
                s_list.append(s_past[0])

                s_past = np.dot(weekly_transition_matrix,s_past)

            return  {'point forecast' : point_forecast,
                     'l_list' : l_list, 'b_list' : b_list, 's_list' : s_list}


   
    #setting the experiment
    mlflow.set_experiment("ETS_Exog_B_A")
    
    # Useful for multiple runs (only doing one run in this sample notebook)    
    with mlflow.start_run():

        #Defining Starting Parameters
        #Optimal Starting parameters after running the starting parameters calculated by the Hyndman method for two iterations
        Starting_Parameters_optimal = [ 2.32625532e-01,  1.00000000e-06,  1.41907946e-02,  9.99333847e-01,
                5.55499458e+03,  3.96440052e+01,  1.14589164e+00,  1.18053933e+00,
                8.78903981e-01,  7.82677252e-01,  7.54118200e-01,  7.76617802e-01,
                9.27728973e-01,  1.28533624e-01, -5.34822743e-02, -1.50822221e-01,
                1.44746722e-02,  1.25113251e-02,np.zeros(len(exogen.columns)-5)]
        
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

        #running the model optimization
        res = minimize(model, Starting_Parameters_optimal, args=(np.array(y['revenue']), exog_to_train), 
                       method='L-BFGS-B', bounds = bounds)

        #the fit extracter is run with the optimal values optained from the optimizer (res.x) and the time series y
        fit = fit_extracter(res.x, np.array(y['revenue']), exog_to_train)

        #creating a data frame with the time series as date object and index
        fit_values = pd.DataFrame({'fitted' : fit['point forecast'], 'date' : pd.to_datetime(y.index)})
        fit_values = fit_values.set_index('date')

        #Plotting results
        revenue_CA_1_FOODS_day.index =pd.to_datetime(revenue_CA_1_FOODS_day.index)
        #Plot the fit and the training data set
        #plt.figure(figsize=(15, 5))
        #plt.plot(revenue_CA_1_FOODS_day[:-365], color = 'blue')
        #plt.plot(fit_values, color="green")
        #plt.xlabel("date")
        #plt.ylabel("revenue_CA_1_FOODS")
        #plt.legend(("realization", "fitted"),  
        #               loc="upper left")
        #plt.savefig('fit_total_plot.png')
       
        #mlflow.log_artifact("./fit_total_plot.png")

        #Plot the fitted and training data set fpr the first year
        #plt.figure(figsize=(15, 5))
        #plt.plot(revenue_CA_1_FOODS_day[:366])
        #plt.plot(fit_values[:366], color="green")
        #plt.xlabel("date")
        #plt.ylabel("revenue_CA_1_FOODS")
        #plt.legend(("realization", "fitted"),  
        #               loc="upper left")
        #plt.savefig('fit_1year_plot.png')
        
        #mlflow.log_artifact("./fit_1year_plot.png")

        #extracting the last (most recent) values of the states for forecasting
        l_values = fit['l_list'][len(fit['l_list'])-1:]
        b_values = fit['b_list'][len(fit['b_list'])-1:]
        s_values = fit['s_list'][len(fit['s_list'])-7:]

        #creating a list of all optimal parameters for forecasting
        forecast_parameters = np.concatenate([res.x[0:4],l_values,b_values,s_values,res.x[13:13+len(exogen.columns)]],
                                             axis=None)


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
        #plt.figure(figsize=(15, 5))
        #plt.plot(revenue_CA_1_FOODS_day, color = 'blue')
        #plt.plot(fit_values, color="green")
        #plt.plot(forecasted_values_365, color="red")
        #plt.xlabel("date")
        #plt.ylabel("revenue_CA_1_FOODS")
        #plt.legend(("realization", "fitted","forecast"),  
        #               loc="upper left")
        #plt.savefig('fit_forecast_total_plot.png')
        
        #mlflow.log_artifact("./fit_forecast_total_plot.png")

        #make sure the prediction data set index is a date variable for plotting
        y_predict.index =pd.to_datetime(y_predict.index)

        #Plot the first 31 days of the prediction data and their forecasts
        #plt.figure(figsize=(15, 5))
        #plt.plot(y_predict[:31])
        #plt.plot(forecasted_values_31, color="red")
        #plt.xlabel("date")
        #plt.ylabel("revenue_CA_1_FOODS")
        #plt.legend(("realization", "forecast"),  
        #               loc="upper left")
        # plt.savefig('fit_forecast_31days_plot.png')
        
        #mlflow.log_artifact("./fit_forecast_31days_plot.png")
        
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
        mlflow.log_param("before", before)
        mlflow.log_param("after", after)
                            
        #Saving optimal Parameters as csv artifact
        Optimum_Parameters = pd.DataFrame(res.x)
        Optimum_Parameters.to_csv('Optimum_Parameters.csv') 
        mlflow.log_artifact("./Optimum_Parameters.csv")
        
        

        #mlflow.log_model(model, "ETS_Exogen")

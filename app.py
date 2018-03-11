import os
from flask import Flask, request, jsonify

import numpy as np
import pandas as pd
from pyramid.arima import auto_arima, ARIMA

from scipy import stats



app = Flask(__name__)


# Inverse Box-Cox Transformation Function
def invboxcox(y,lmbda):
    if lmbda == 0:
        return(np.exp(y))
    else:
        return(np.exp(np.log(lmbda*y+1)/lmbda))


def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100    

@app.route('/')
def hello():
    return 'Fast Food - Forecasting!'

@app.route('/api/v1/sales')
def forecasting_sales():

	period = request.args.get('period') 
	data = pd.read_excel('notebooks/data/food-sp.xlsx')
	variavel = 'VENDA'

	data.index = data['DATA']

	interval = 96 - int(period)
	df_train = data.iloc[1:interval,]
	df_test = data.iloc[interval:96,] 

	df_train[variavel+'_box'], lmbda = stats.boxcox(df_train[variavel])

	# model = auto_arima(df_train[variavel+'_box'], 
 #                    n_fits=10,
 #                    start_p=0, 
 #                    start_q=0, 
 #                    max_p=5, 
 #                    max_q=5, 
 #                    m=20,
 #                    start_P=0, 
 #                    d=1, 
 #                    D=1, 
 #                    trace=True,
 #                    stationary=False,
 #                    error_action='ignore',
 #                    suppress_warnings=True,
 #                    stepwise=True)

	model = ARIMA(callback=None, disp=0, maxiter=50, method=None, order=(1, 1, 1),
	   out_of_sample_size=0, scoring='mse', scoring_args={},
	   seasonal_order=(2, 1, 1, 20), solver='lbfgs', start_params=None,
	   suppress_warnings=True, transparams=True, trend='c')

	model.fit(df_train[variavel+'_box'])
	# model.summary()

	forecast = model.predict(n_periods=int(period))

	y_pred = invboxcox(forecast,lmbda)
	y_true = df_test[variavel].values

	acuracia = round(100 - mean_absolute_percentage_error(y_true , y_pred),0)

	retorno = {'acuracia' : acuracia, 'real' : y_true.tolist(), 'previsto' : y_pred.tolist()}

	return jsonify(retorno)

if __name__ == '__main__':
    # Bind to PORT if defined, otherwise default to 5000.
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
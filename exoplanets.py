import numpy as np
from matplotlib import pyplot as plt
import csv
from numpy import random
#from sklearn.linear_model import LinearRegression
#from sklearn.metrics import mean_squared_error, r2_score

def get_dataset(filename):
    datalist = []
    with open(filename) as f:
        dictReader = csv.DictReader(f)
        datalist = []
        for row in dictReader:
            planet = {}
            
            if(row['pl_orbper'] != None or row['pl_bmasse'] != None):
                planet['pl_name'] = row['pl_name']
                try:
                    planet['pl_orbper'] = float(row['pl_orbper'].strip())
                    planet['pl_bmasse'] = float(row['pl_bmasse'].strip())
                    planet['st_teff'] = float(row['st_teff'].strip()) / 1000
                    planet['pl_eqt'] = float(row['pl_eqt'].strip()) / 1000
                    datalist.append(planet)
                except ValueError:
                    #print(row['pl_orbper'].strip())
                    #print(row['pl_bmasse'].strip())
                    continue
    
    dataset = np.array(datalist)
    
    return dataset


#y - the prediction to study against
#keys - the keys to study the relationship between y
def regression(dataset, betas, keys, y):
    tot = 0
    n = 0
    for row in dataset:
        n += 1
        temp_sum = 0
        temp_sum += betas[0]
        beta_index = 1
        for key in keys:
            temp_sum += (row[key] * betas[beta_index])
        
        temp_sum -= row[y]

        tot += (temp_sum ** 2)

    return tot / n


def gradient_descent(dataset, keys, betas, y):
    beta_derivs = []
    
    
    for i in range(len(betas)):
        
        mse = 0
        n = 0
        if(i - 1 == -1):
            for row in dataset:
                n += 1
                temp = 0
                temp += betas[0]
                #(actual - predicted)^2
                col_index = 0
                for key in keys:
                    col_index += 1
                    temp += float(betas[col_index] * row[key])
                temp -= row[y]
                
                mse += temp
        
        else:
            for row in dataset:
                n += 1
                temp = 0
                temp += betas[0]
                #(actual - predicted)^2
                col_index = 0
                for key in keys:
                    col_index += 1
                    temp += float(betas[col_index] * row[key])
                temp -= row[y]
                try:
                    mse += (temp * row[keys[i - 1]])     
                except IndexError:
                    print("sad")
                    continue
                
        
        #print(mse * (2/n))
        beta_derivs.append(float((mse * 2) / n))
        
    return beta_derivs



def iterate_gradient(dataset, cols, betas, T, eta, y):
    #print T, mse, b0, b1, b2..
    
    curr_betas = []
    curr_par_deriv = gradient_descent(dataset, cols, betas, y)
    
    for beta in betas:
        curr_betas.append(beta)
        
    for i in range(T):
        curr_par_deriv = gradient_descent(dataset, cols, curr_betas, y)

        output = str(i + 1)
        x = 0
        for beta in curr_betas:
            curr_betas[x] = beta - (eta * curr_par_deriv[x])
            x += 1
        
        output = output + " " + str("%.2f" % round(regression(dataset, curr_betas, cols, y), 2))
        
        for z in range(len(curr_betas)):
            output = output + " " + str("%.2f" % round(curr_betas[z], 2))        
        
        print(output)
    return curr_betas


def predict(dataset, features, betas):
    """

    INPUT: 
        dataset - the body fat n by m+1 array
        cols    - a list of feature indices to learn.
                  For example, [1,8] refers to density and abdomen.
        features- a list of observed values

    RETURNS:
        The predicted y value
    """
    pred_y = betas[0]
    
    i = 1


    for feature in features:
        pred_y += feature * betas[i]
        i += 1

    return pred_y



data = get_dataset('./exoplanet_data.csv')
plt.figure()

stellar_temp = []
pl_temp = []

for row in data:
    
    stellar_temp.append(row['st_teff'])
    pl_temp.append(row['pl_eqt'])

beta_vals = iterate_gradient(data, ['pl_eqt'], [1,-1], 400, 1e-1, 'st_teff')

y_vals = []
for row in data:
    y_vals.append(predict(data, [row['pl_eqt']], beta_vals))

#regression_model = LinearRegression()
# Fit the data(train the model)
#regression_model.fit(pl_temp, stellar_temp)
# Predict
#y_predicted = regression_model.predict(pl_temp)

#plt.plot()



plt.scatter(pl_temp, stellar_temp)
plt.plot(pl_temp, y_vals)
plt.xlabel('pl_temp')
plt.ylabel('st_temp')
#plt.show()









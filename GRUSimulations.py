import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import GRUModel
import tensorflow as tf
from sklearn.metrics import root_mean_squared_error
from tqdm import tqdm
import time

start = time.time()    #Starting the time count

#PPreparing synthetic data of geodesic curves to simulate trajectories and train GRU

curves = []

x = np.linspace(0, 1, 1000) 

n_values = np.linspace(0.1, 10, 100)

for n in n_values: 
    epsilon_x = np.random.normal(0, 0.002, len(x)) #Adding random noise to the function
    y = x**(n) + epsilon_x
    curves.append(y)
    
#Saving the synthetic data in a dataframe

df_curves = pd.DataFrame(
    np.array(curves).T,
    index = x,
    columns = [f'n_{n:.2f}' for n in n_values]
)

#Applying GRU Model in the synthetic data and trying to predict/reconstruct the desired route

X = df_curves.values.T.astype(np.float32)

X = (X - X.mean(axis=1, keepdims=True)) / \
    (X.std(axis=1, keepdims=True) + 1e-8)    #Curves normalization

X_c = X[:, :, np.newaxis]  #(100, 1000, 1)

#Input and Target
X_in  = X_c[:, :-1, :]              
Y_out = X_c[:,  1:, :]              

X_in  = tf.transpose(X_in,  [1, 0, 2])   
Y_out = tf.transpose(Y_out, [1, 0, 2])  

X_in  = tf.cast(X_in,  tf.float32)
Y_out = tf.cast(Y_out, tf.float32)

model = GRUModel.GRUModel(num_inputs = 1, num_hiddens = 32, num_outputs = 1)

for epoch in tqdm(range(200), desc = "Training GRU"): 
    loss = GRUModel.train_step(model, X_in, Y_out)
    if epoch % 20 == 0:
        tqdm.write(f"Epoch {epoch} | Loss: {loss.numpy(): .6f}")


Y_pred = model(X_in)  #Reconstructing the route based on training

idx = 99 #Target Prediction

#Calculating the error between the real curve and the prediction to evaluate the performance of GRU

rmse_seq = [] 

for i in range(Y_out.shape[0]): 
    y_true = tf.reshape(Y_out[i], (-1,))
    y_pred = tf.reshape(Y_pred[i], (-1,))
    
    rmse = root_mean_squared_error(
        y_true.numpy(),
        y_pred.numpy()
    )
    rmse_seq.append(rmse)

df_rmse_pred = pd.Series(
    rmse_seq, 
    name='RMSE'
    )

end = time.time() #Ending time counting

total_time = end- start

#Ploting the synthetic curve and the prediction

plt.figure(figsize=(7, 5))
plt.plot(Y_out[:, idx, 0], label='Synthetic Data')
plt.plot(Y_pred[:, idx, 0], '--', label='GRU Reconstruction')
plt.legend()
plt.grid(True)

print(f"Total execution time: {total_time: .2f} seconds.")

#Showing the RMSE calculated of the desired curve

print(df_rmse_pred)

plt.show()
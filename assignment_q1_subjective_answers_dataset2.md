### Task 1: Dataset 2










    (40, 2)
    Theta matrix: [3.9507064  2.68246893]
    Min Loss: 0.5957541565733318
    



### Plot of Loss vs Epoch for Full Batch Gradient Descent





    
![png](ML__A2_task1_dataset2_files/ML__A2_task1_dataset2_7_0.png)
    




    


    
![png](ML__A2_task1_dataset2_files/ML__A2_task1_dataset2_8_1.png)
    


### Contour Plot for Full Batch Gradient Descent





    
![png](ML__A2_task1_dataset2_files/ML__A2_task1_dataset2_10_0.png)
    


    C:\Users\seema\AppData\Local\Temp\ipykernel_16872\3438717558.py:80: DeprecationWarning: Starting with ImageIO v3 the behavior of this function will switch to that of iio.v3.imread. To keep the current behavior (and make this warning disappear) use `import imageio.v2 as imageio` or call `imageio.v2.imread` directly.
      image = imageio.imread(filename)
    

    GIF saved permanently at: results\trajectory_bgd_no_momentum.gif
    

### Average Convergence Steps for Full Batch Gradient Descent






     Average convergence steps across 5 runs: 554.6
    




    



### Stochastic Gradient Descent



### Plot of Loss vs Epoch for Stochastic Gradient Descent





    
![png](ML__A2_task1_dataset2_files/ML__A2_task1_dataset2_17_1.png)
    


### Contour Plot for Stochastic gradient Descent




    
![png](ML__A2_task1_dataset2_files/ML__A2_task1_dataset2_19_0.png)
    


    C:\Users\seema\AppData\Local\Temp\ipykernel_16872\235109145.py:80: DeprecationWarning: Starting with ImageIO v3 the behavior of this function will switch to that of iio.v3.imread. To keep the current behavior (and make this warning disappear) use `import imageio.v2 as imageio` or call `imageio.v2.imread` directly.
      image = imageio.imread(filename)
    

    GIF saved permanently at: results\trajectory_sgd_no_momentum.gif
    

### Average Convergence Steps for Stochastic Gradient Descent



     Average convergence steps across 5 runs: 688.0
    








### Gradient Descent with momentum



### Plot of Loss Vs Epochs for Full Batch Gradient Descent with momentum=0.9





    
![png](ML__A2_task1_dataset2_files/ML__A2_task1_dataset2_26_0.png)
    




    


    
![png](ML__A2_task1_dataset2_files/ML__A2_task1_dataset2_27_1.png)
    


### Contour Plot for Full Batch Gradient Descent with momentum = 0.9





    
![png](ML__A2_task1_dataset2_files/ML__A2_task1_dataset2_29_0.png)
    


    C:\Users\seema\AppData\Local\Temp\ipykernel_16872\3835177526.py:82: DeprecationWarning: Starting with ImageIO v3 the behavior of this function will switch to that of iio.v3.imread. To keep the current behavior (and make this warning disappear) use `import imageio.v2 as imageio` or call `imageio.v2.imread` directly.
      image = imageio.imread(filename)
    

    GIF saved permanently at: results\trajectory_bgd_momentum.gif
    





<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>epoch</th>
      <th>t0</th>
      <th>t1</th>
      <th>t0_grad</th>
      <th>t1_grad</th>
      <th>velocity_t0</th>
      <th>velocity_t1</th>
      <th>loss</th>
    </tr>
    <tr>
      <th>iteration</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1.056164</td>
      <td>1.005630</td>
      <td>-5.616435</td>
      <td>-0.562997</td>
      <td>-0.056164</td>
      <td>-0.005630</td>
      <td>9.355592</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>1.161763</td>
      <td>1.016387</td>
      <td>-5.505060</td>
      <td>-0.568954</td>
      <td>-0.105599</td>
      <td>-0.010757</td>
      <td>9.040090</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>1.309758</td>
      <td>1.031868</td>
      <td>-5.295684</td>
      <td>-0.580046</td>
      <td>-0.147996</td>
      <td>-0.015481</td>
      <td>8.463638</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>1.492977</td>
      <td>1.051754</td>
      <td>-5.002316</td>
      <td>-0.595334</td>
      <td>-0.183219</td>
      <td>-0.019887</td>
      <td>7.692512</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5</td>
      <td>1.704267</td>
      <td>1.075790</td>
      <td>-4.639246</td>
      <td>-0.613806</td>
      <td>-0.211290</td>
      <td>-0.024036</td>
      <td>6.797230</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>96</th>
      <td>96</td>
      <td>3.932266</td>
      <td>2.677953</td>
      <td>-0.038730</td>
      <td>0.000973</td>
      <td>-0.001242</td>
      <td>0.000778</td>
      <td>0.596133</td>
    </tr>
    <tr>
      <th>97</th>
      <td>97</td>
      <td>3.933745</td>
      <td>2.677250</td>
      <td>-0.036115</td>
      <td>0.000271</td>
      <td>-0.001479</td>
      <td>0.000703</td>
      <td>0.596087</td>
    </tr>
    <tr>
      <th>98</th>
      <td>98</td>
      <td>3.935406</td>
      <td>2.676621</td>
      <td>-0.033038</td>
      <td>-0.000424</td>
      <td>-0.001661</td>
      <td>0.000629</td>
      <td>0.596035</td>
    </tr>
    <tr>
      <th>99</th>
      <td>99</td>
      <td>3.937197</td>
      <td>2.676066</td>
      <td>-0.029610</td>
      <td>-0.001103</td>
      <td>-0.001791</td>
      <td>0.000555</td>
      <td>0.595984</td>
    </tr>
    <tr>
      <th>100</th>
      <td>100</td>
      <td>3.939069</td>
      <td>2.675584</td>
      <td>-0.025934</td>
      <td>-0.001756</td>
      <td>-0.001871</td>
      <td>0.000482</td>
      <td>0.595935</td>
    </tr>
  </tbody>
</table>
<p>100 rows × 8 columns</p>
</div>


### Average Convergence Steps for Full Batch Gradient Descent with momentum= 0.9




     Average convergence steps across 5 runs: 37.6
    




    



### Stochastic Gradient Descent with momentum=0.9



    


    
![png](ML__A2_task1_dataset2_files/ML__A2_task1_dataset2_36_1.png)
    





    
![png](ML__A2_task1_dataset2_files/ML__A2_task1_dataset2_37_0.png)
    


    C:\Users\seema\AppData\Local\Temp\ipykernel_16872\1241512859.py:82: DeprecationWarning: Starting with ImageIO v3 the behavior of this function will switch to that of iio.v3.imread. To keep the current behavior (and make this warning disappear) use `import imageio.v2 as imageio` or call `imageio.v2.imread` directly.
      image = imageio.imread(filename)
    

    GIF saved permanently at: results\trajectory_sgd_momentum_alpha_0.01.gif
    

We observe that stochastic gradient descent with momentum (momentum = 0.9) and a learning rate of 0.01 fails to satisfy the convergence criterion even after 100000 epochs. This happens because the stochastic nature of the algorithm introduces high variance in gradient estimates, causing noisy and fluctuating updates. With a relatively large learning rate, these fluctuations get amplified through the accumulated momentum term, leading to oscillations around the optimum rather than stable convergence. We have decided to reduce the learning rate in order to observe convergence.

### With learning rate=0.001
### Plot of Loss vs Epochs for Stochastic Gradient Descent with momentum = 0.9


    


    
![png](ML__A2_task1_dataset2_files/ML__A2_task1_dataset2_40_1.png)
    


### Contour Plot for Stochastic Gradient Descent with momentum = 0.9




    
![png](ML__A2_task1_dataset2_files/ML__A2_task1_dataset2_42_0.png)
    


    C:\Users\seema\AppData\Local\Temp\ipykernel_16872\3073285957.py:82: DeprecationWarning: Starting with ImageIO v3 the behavior of this function will switch to that of iio.v3.imread. To keep the current behavior (and make this warning disappear) use `import imageio.v2 as imageio` or call `imageio.v2.imread` directly.
      image = imageio.imread(filename)
    

    GIF saved permanently at: results\trajectory_sgd_momentum_alpha_0.001.gif
    

### Average Convergence Steps for Stochastic Gradient Descent with momentum =0.9



    Average convergence steps across 5 runs: 328.0
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>t0</th>
      <th>t1</th>
      <th>t0_grad</th>
      <th>t1_grad</th>
      <th>velocity_t0</th>
      <th>velocity_t1</th>
      <th>epoch</th>
    </tr>
    <tr>
      <th>iteration</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>1.008050</td>
      <td>1.007873</td>
      <td>-8.050039</td>
      <td>-7.873124</td>
      <td>-0.008050</td>
      <td>-0.007873</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.023896</td>
      <td>1.015811</td>
      <td>-8.600670</td>
      <td>-0.852236</td>
      <td>-0.015846</td>
      <td>-0.007938</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.041995</td>
      <td>1.021278</td>
      <td>-3.838243</td>
      <td>1.677717</td>
      <td>-0.018099</td>
      <td>-0.005467</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.063412</td>
      <td>1.021863</td>
      <td>-5.127123</td>
      <td>4.334577</td>
      <td>-0.021417</td>
      <td>-0.000585</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1.087012</td>
      <td>1.021909</td>
      <td>-4.325634</td>
      <td>0.480409</td>
      <td>-0.023601</td>
      <td>-0.000046</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>596</th>
      <td>3.929792</td>
      <td>2.646516</td>
      <td>0.665943</td>
      <td>-0.332931</td>
      <td>-0.001286</td>
      <td>-0.001677</td>
      <td>14</td>
    </tr>
    <tr>
      <th>597</th>
      <td>3.929688</td>
      <td>2.648132</td>
      <td>1.261889</td>
      <td>-0.106660</td>
      <td>0.000104</td>
      <td>-0.001616</td>
      <td>14</td>
    </tr>
    <tr>
      <th>598</th>
      <td>3.930444</td>
      <td>2.649684</td>
      <td>-0.849679</td>
      <td>-0.097599</td>
      <td>-0.000756</td>
      <td>-0.001552</td>
      <td>14</td>
    </tr>
    <tr>
      <th>599</th>
      <td>3.933884</td>
      <td>2.649711</td>
      <td>-2.760366</td>
      <td>1.369416</td>
      <td>-0.003441</td>
      <td>-0.000027</td>
      <td>14</td>
    </tr>
    <tr>
      <th>600</th>
      <td>3.937871</td>
      <td>2.649042</td>
      <td>-0.890447</td>
      <td>0.694119</td>
      <td>-0.003987</td>
      <td>0.000669</td>
      <td>15</td>
    </tr>
  </tbody>
</table>
<p>600 rows × 7 columns</p>
</div>


### Observations on Vanilla Gradient Descent

When computing the average number of steps required for each method to converge, it is observed that the stochastic gradient descent (SGD) method takes more steps than full-batch gradient descent (BGD) . Although SGD may require fewer epochs to converge, each epoch involves updating the parameters after every sample, making the total number of steps (i.e., parameter updates) much higher. This frequent updating causes the loss function to fluctuate significantly around the minimum due to the high variance in gradient estimates from individual samples.

In contrast, the full-batch gradient descent (BGD) method updates the parameters only once per epoch after processing the entire dataset, resulting in fewer total steps and a smoother, more stable convergence curve. This reduces fluctuations in the loss function and allows a more consistent approach toward the global minimum.

From the experimental results averaged across 5 runs:

The average number of steps for full-batch gradient descent to converge is ≈ 554.6 steps.

The average number of steps for stochastic gradient descent to converge is ≈ 688 steps.

However, considering computation time, SGD generally converges faster in practice, as each step requires processing only a single sample, making individual updates significantly quicker than those in BGD. Thus, while BGD is more stable, SGD achieves practical convergence more rapidly due to its frequent but lightweight updates.

### Observations on Gradient Descent with Momentum

When computing the average number of steps required for each method with momentum to converge, it is observed that adding momentum significantly reduces the total number of steps compared to the vanilla methods. The momentum term incorporates a fraction of the previous update into the current update, which helps the optimization maintain direction and reduces oscillations in the loss function. This effect, similar to inertia in physics, allows the optimizer to move more smoothly and consistently toward the minimum, resulting in faster convergence.

The average number of steps for full-batch gradient descent with momentum to converge is ≈ 37.6 steps much lesser than steps required for vanilla BGD (554.6).

However, when using SGD with a high momentum coefficient (0.9) and the same learning rate as the vanilla cases (0.01), the optimization failed to converge even after 100,000 epochs. This occurs because the combination of a large learning rate and strong momentum causes excessively large parameter updates. The optimizer overshoots the minimum repeatedly, leading to divergence and instability in the loss function. By reducing the learning rate to 0.001, the updates became smaller and more controlled, allowing the benefits of momentum to accelerate convergence without destabilizing the optimization. With this adjustment, the average number of steps reduced to 328 compared to vanilla SGD (688)—demonstrating that momentum can dramatically improve convergence speed when paired with an appropriate learning rate.

Observation on Dataset Scaling

It is observed that all optimization methods required more steps to converge on Dataset 1 compared to Dataset 2. This difference arises primarily due to the scale of the input features and target values. In Dataset 1, the inputs range from -20 to 20 and the target values are scaled by 100, producing large gradients during optimization. These large gradients cause the parameter updates to be more volatile, leading to oscillations around the minimum and requiring more steps for the optimizer to stabilize. In contrast, Dataset 2 has inputs ranging from -1 to 1 and smaller target values, resulting in smaller and more stable gradients. This allows the optimizer to take proportionate steps toward convergence, reducing the total number of steps required. Thus, feature and target scaling significantly affect the convergence behavior of gradient-based optimization methods.

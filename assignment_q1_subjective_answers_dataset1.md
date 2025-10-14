### Task 1: Dataset 1




    (40, 2)
    Theta matrix: [ 0.9507064  99.98412345]
    Min Loss: 0.5957541565733389
    




### Plot of Loss vs Epoch for Full Batch Gradient Descent





    
![png](ML__A2_task1_newdataset1_files/ML__A2_task1_newdataset1_7_0.png)
    




    


    
![png](ML__A2_task1_newdataset1_files/ML__A2_task1_newdataset1_8_1.png)
    


### Contour Plot for Full Batch Gradient Descent



    
![png](ML__A2_task1_newdataset1_files/ML__A2_task1_newdataset1_10_0.png)
    


    C:\Users\seema\AppData\Local\Temp\ipykernel_23628\2621361301.py:82: DeprecationWarning: Starting with ImageIO v3 the behavior of this function will switch to that of iio.v3.imread. To keep the current behavior (and make this warning disappear) use `import imageio.v2 as imageio` or call `imageio.v2.imread` directly.
      image = imageio.imread(filename)
    

    GIF saved permanently at: results\trajectory_bgd_no_momentum_dataset1.gif
    

### Average Convergence Steps for Full Batch Gradient Descent



     Average convergence steps across 5 runs: 1753.8
    




    



### Stochastic Gradient Descent




### Plot of Loss vs Epoch for Stochastic Gradient Descent



    


    
![png](ML__A2_task1_newdataset1_files/ML__A2_task1_newdataset1_17_1.png)
    




    


    
![png](ML__A2_task1_newdataset1_files/ML__A2_task1_newdataset1_18_1.png)
    


### Contour Plot for Stochastic Gradient Descent



    
![png](ML__A2_task1_newdataset1_files/ML__A2_task1_newdataset1_20_0.png)
    


    C:\Users\seema\AppData\Local\Temp\ipykernel_23628\687517133.py:82: DeprecationWarning: Starting with ImageIO v3 the behavior of this function will switch to that of iio.v3.imread. To keep the current behavior (and make this warning disappear) use `import imageio.v2 as imageio` or call `imageio.v2.imread` directly.
      image = imageio.imread(filename)
    

    GIF saved permanently at: results\trajectory_sgd_no_momentum_dataset1.gif
    

### Average Convergence Steps for Stochastic Gradient Descent



     Average convergence steps across 4 runs: 13010.0
    




    



###  Gradient Descent with momentum



### Plot of Loss Vs Epochs for Full Batch Gradient Descent with momentum=0.8





    
![png](ML__A2_task1_newdataset1_files/ML__A2_task1_newdataset1_27_0.png)
    





    
![png](ML__A2_task1_newdataset1_files/ML__A2_task1_newdataset1_28_1.png)
    


### Contour Plot for Full Batch Gradient Descent with momentum = 0.8



    
![png](ML__A2_task1_newdataset1_files/ML__A2_task1_newdataset1_30_0.png)
    


    C:\Users\seema\AppData\Local\Temp\ipykernel_23628\2694275616.py:82: DeprecationWarning: Starting with ImageIO v3 the behavior of this function will switch to that of iio.v3.imread. To keep the current behavior (and make this warning disappear) use `import imageio.v2 as imageio` or call `imageio.v2.imread` directly.
      image = imageio.imread(filename)
    

    GIF saved permanently at: results\trajectory_bgd_momentum_dataset1.gif
    






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
      <td>0.664581</td>
      <td>26.010887</td>
      <td>335.418915</td>
      <td>-25010.886719</td>
      <td>0.335419</td>
      <td>-25.010887</td>
      <td>1.237849e+06</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>0.146225</td>
      <td>64.709747</td>
      <td>250.020752</td>
      <td>-18690.148438</td>
      <td>0.518356</td>
      <td>-38.698860</td>
      <td>6.912502e+05</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>-0.386347</td>
      <td>104.579033</td>
      <td>117.887115</td>
      <td>-8910.194336</td>
      <td>0.532572</td>
      <td>-39.869286</td>
      <td>1.571040e+05</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>-0.794164</td>
      <td>135.308914</td>
      <td>-18.239901</td>
      <td>1165.543823</td>
      <td>0.407818</td>
      <td>-30.729885</td>
      <td>2.690574e+03</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5</td>
      <td>-0.997262</td>
      <td>150.961258</td>
      <td>-123.156609</td>
      <td>8931.569336</td>
      <td>0.203097</td>
      <td>-15.652338</td>
      <td>1.578609e+05</td>
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
      <th>496</th>
      <td>496</td>
      <td>0.942336</td>
      <td>99.984009</td>
      <td>-0.016524</td>
      <td>-0.000324</td>
      <td>-0.000086</td>
      <td>-0.000002</td>
      <td>5.958240e-01</td>
    </tr>
    <tr>
      <th>497</th>
      <td>497</td>
      <td>0.942422</td>
      <td>99.984009</td>
      <td>-0.016352</td>
      <td>-0.000616</td>
      <td>-0.000085</td>
      <td>-0.000003</td>
      <td>5.958226e-01</td>
    </tr>
    <tr>
      <th>498</th>
      <td>498</td>
      <td>0.942506</td>
      <td>99.984009</td>
      <td>-0.016181</td>
      <td>-0.000905</td>
      <td>-0.000084</td>
      <td>-0.000003</td>
      <td>5.958212e-01</td>
    </tr>
    <tr>
      <th>499</th>
      <td>499</td>
      <td>0.942590</td>
      <td>99.984009</td>
      <td>-0.016012</td>
      <td>-0.001191</td>
      <td>-0.000084</td>
      <td>-0.000004</td>
      <td>5.958199e-01</td>
    </tr>
    <tr>
      <th>500</th>
      <td>500</td>
      <td>0.942672</td>
      <td>99.984016</td>
      <td>-0.015845</td>
      <td>-0.001474</td>
      <td>-0.000083</td>
      <td>-0.000004</td>
      <td>5.958185e-01</td>
    </tr>
  </tbody>
</table>
<p>500 rows × 8 columns</p>
</div>


### Average Convergence Steps for Full Batch Gradient Descent with momentum= 0.8




     Average convergence steps across 5 runs: 282.4
    




 



### Stochastic Gradient Descent with momentum=0.8




    
![png](ML__A2_task1_newdataset1_files/ML__A2_task1_newdataset1_37_0.png)
    




    
![png](ML__A2_task1_newdataset1_files/ML__A2_task1_newdataset1_38_0.png)
    


    C:\Users\seema\AppData\Local\Temp\ipykernel_23628\2807771302.py:82: DeprecationWarning: Starting with ImageIO v3 the behavior of this function will switch to that of iio.v3.imread. To keep the current behavior (and make this warning disappear) use `import imageio.v2 as imageio` or call `imageio.v2.imread` directly.
      image = imageio.imread(filename)
    

    GIF saved permanently at: results\trajectory_sgd_momentum_dataset1_alpha0.001.gif
    GIF saved permanently at: results\trajectory_sgd_momentum_dataset1_alpha0.001.gif
    

We observe that stochastic gradient descent with momentum (momentum = 0.8) and a learning rate of 0.001 fails to satisfy the convergence criterion even after 100000 epochs. This happens because the stochastic nature of the algorithm introduces high variance in gradient estimates, causing noisy and fluctuating updates. With a relatively large learning rate, these fluctuations get amplified through the accumulated momentum term, leading to oscillations around the optimum rather than stable convergence. We have decided to reduce the learning rate in order to observe convergence.

### Considering learning rate 0.00001
### Plot of Loss vs Epochs for Stochastic Gradient Descent with momentum = 0.8



    
![png](ML__A2_task1_newdataset1_files/ML__A2_task1_newdataset1_41_0.png)
    




    
![png](ML__A2_task1_newdataset1_files/ML__A2_task1_newdataset1_42_0.png)
    


### Contour Plot for Stochastic Gradient Descent with momentum = 0.8



    
![png](ML__A2_task1_newdataset1_files/ML__A2_task1_newdataset1_44_0.png)
    


    C:\Users\seema\AppData\Local\Temp\ipykernel_23628\3824559797.py:82: DeprecationWarning: Starting with ImageIO v3 the behavior of this function will switch to that of iio.v3.imread. To keep the current behavior (and make this warning disappear) use `import imageio.v2 as imageio` or call `imageio.v2.imread` directly.
      image = imageio.imread(filename)
    

    GIF saved permanently at: results\trajectory_sgd_momentum_alpha0.00001.gif
    

### Average Convergence Steps for Stochastic Gradient Descent with momentum =0.8


    Average convergence steps across 5 runs: 33672.0
    




   






<div>

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
      <td>1.038711</td>
      <td>1.757207</td>
      <td>-3871.109131</td>
      <td>-75720.679688</td>
      <td>-3.871109e-02</td>
      <td>-0.757207</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.073595</td>
      <td>2.370732</td>
      <td>-391.537537</td>
      <td>-775.944824</td>
      <td>-3.488425e-02</td>
      <td>-0.613525</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.084428</td>
      <td>3.010823</td>
      <td>1707.497803</td>
      <td>-14927.129883</td>
      <td>-1.083242e-02</td>
      <td>-0.640091</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.060319</td>
      <td>4.077066</td>
      <td>3277.479004</td>
      <td>-55416.984375</td>
      <td>2.410885e-02</td>
      <td>-1.066243</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1.036758</td>
      <td>4.939553</td>
      <td>427.360687</td>
      <td>-949.262207</td>
      <td>2.356069e-02</td>
      <td>-0.862487</td>
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
      <th>39996</th>
      <td>0.928102</td>
      <td>99.985199</td>
      <td>0.609635</td>
      <td>-6.095605</td>
      <td>-1.460987e-05</td>
      <td>-0.000183</td>
      <td>999</td>
    </tr>
    <tr>
      <th>39997</th>
      <td>0.928101</td>
      <td>99.985367</td>
      <td>1.248795</td>
      <td>-2.111063</td>
      <td>8.000488e-07</td>
      <td>-0.000168</td>
      <td>999</td>
    </tr>
    <tr>
      <th>39998</th>
      <td>0.928109</td>
      <td>99.985519</td>
      <td>-0.839251</td>
      <td>-1.928018</td>
      <td>-7.752472e-06</td>
      <td>-0.000154</td>
      <td>999</td>
    </tr>
    <tr>
      <th>39999</th>
      <td>0.928143</td>
      <td>99.985359</td>
      <td>-2.825266</td>
      <td>28.032263</td>
      <td>-3.445464e-05</td>
      <td>0.000157</td>
      <td>999</td>
    </tr>
    <tr>
      <th>40000</th>
      <td>0.928181</td>
      <td>99.985077</td>
      <td>-0.991531</td>
      <td>15.458310</td>
      <td>-3.747902e-05</td>
      <td>0.000281</td>
      <td>1000</td>
    </tr>
  </tbody>
</table>
<p>40000 rows × 7 columns</p>
</div>


### Observations on Vanilla Gradient Descent

When computing the average number of steps required for each method to converge, it is observed that the stochastic gradient descent (SGD) method takes more steps than full-batch gradient descent (BGD) . Although SGD may require fewer epochs to converge, each epoch involves updating the parameters after every sample, making the total number of steps (i.e., parameter updates) much higher. This frequent updating causes the loss function to fluctuate significantly around the minimum due to the high variance in gradient estimates from individual samples.

In contrast, the full-batch gradient descent (BGD) method updates the parameters only once per epoch after processing the entire dataset, resulting in fewer total steps and a smoother, more stable convergence curve. This reduces fluctuations in the loss function and allows a more consistent approach toward the global minimum.

From the experimental results averaged across 5 runs:

The average number of steps for full-batch gradient descent to converge is ≈ 1753 steps.

The average number of steps for stochastic gradient descent to converge is ≈ 688 steps.

However, considering computation time, SGD generally converges faster in practice, as each step requires processing only a single sample, making individual updates significantly quicker than those in BGD. Thus, while BGD is more stable, SGD achieves practical convergence more rapidly due to its frequent but lightweight updates.

### Observations on Gradient Descent with Momentum

When computing the average number of steps required for each method with momentum to converge, it is observed that adding momentum significantly reduces the total number of steps compared to the vanilla methods. The momentum term incorporates a fraction of the previous update into the current update, which helps the optimization maintain direction and reduces oscillations in the loss function. This effect, similar to inertia in physics, allows the optimizer to move more smoothly and consistently toward the minimum, resulting in faster convergence.

The average number of steps for full-batch gradient descent with momentum to converge is ≈ 282.4 steps, much less than the steps required for vanilla BGD (1753).

When applying SGD with momentum on this dataset, it is observed that using a high momentum coefficient (0.8) with the same learning rate as the vanilla cases (0.001) resulted in failure to converge even after 100,000 epochs. This happens because the combination of a strong momentum term and a relatively large learning rate causes the parameter updates to overshoot the minimum repeatedly, leading to divergence and instability in the loss function.

By reducing the learning rate to 0.00001, the updates became smaller and more controlled, allowing the optimizer to stabilize and benefit from momentum. However, unlike previous cases, the average number of steps required to converge increased to 33,672 compared to vanilla SGD (13,010 steps). This indicates that, for this dataset, the momentum term helped stabilize convergence but required more steps due to smaller learning rate adjustments. It demonstrates that while momentum can accelerate convergence in some scenarios, its effectiveness depends critically on the learning rate and the characteristics of the dataset.

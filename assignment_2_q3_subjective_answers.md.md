## Question 3: Working with Autoregressive Modeling [2 Marks]

* [2 marks] Consider the [Daily Temperatures dataset](https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-temperatures.csv) from Australia. This is a dataset for a forecasting task. That is, given temperatures up to date (or period) T, design a forecasting (autoregressive) model to predict the temperature on date T+1. You can refer to [link 1](https://en.wikipedia.org/wiki/Autoregressive_model), [link 2](https://machinelearningmastery.com/autoregression-models-time-series-forecasting-python/) for more information on autoregressive models. Use linear regression as your autoregressive model. Plot the fit of your predictions vs the true values and report the RMSE obtained. A demonstration of the plot is given below.

---
![alt text](image.png)


### Observations

* **Strong Baseline Performance:** Our model is a solid baseline, accurately tracking the main temperature patterns. The low RMSE score of around 2.5°C shows that its predictions are typically off by only a small amount.

* **Good Generalization:** The model generalizes well to new data. Its consistent performance on both the training and test sets, confirmed by their similar RMSE scores, proves that it has learned the underlying patterns without overfitting.

* **Smoothing of Extremes:** The model tends to smooth out the data, rarely predicting the most extreme temperature highs and lows. It captures the overall trend but is less sensitive to sharp, infrequent spikes.
# Higher Order Effects in Derivative Regularization for Adversarial Robustness

Adversarial attacks remain one of the main flaws of deep neural networks today. Besides adversarial training, some researchers have identified derivative regularisation (first and second order) to be another usefool tool for both understanding and tackling adversarial robustness. However, these methods do not quite reach the same level of effectiveness as adversarial training. 

In this repository, we explore the role of mixed and higher order effects in regularization and how/if it can further narrow the gap to adversarial training. Furthermore, we explore improving the accuracy of the derivative estimators to higher orders.

The base for our code is https://github.com/F-Salehi/CURE_robustness. However, we changed quite a lot and so this repo has different dependencies. 

To run our code, run `DL_project/CURE_robustness_mod/main.py`. You can set hyper parameters for e.g. regularizer constants, data set and level of estimation accuracy in `CURE_robustness_mod/utils/config.py`.

We recommend you to create a conda or virtual environment from our `CURE_robustness_mod/requirements.txt` with PyTorch version `3.7.1`. The code might still run if you do it differently, but no guarantees. :no_good:

If you still have any questions, feel free to reach out to us. Otherwise, enjoy our code! :raised_hands:

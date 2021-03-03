# MachineLearning

<p align="left">
  <img src="https://www.python.org/static/community_logos/python-logo-master-v3-TM.png" width="150" title="Python 3">
  <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/3/31/NumPy_logo_2020.svg/512px-NumPy_logo_2020.svg.png" width="150" title="NumPY">
   <img src="https://camo.githubusercontent.com/109927a15915074d15313889468aa9aa688de3b9e38cc4359a01f665d351114e/68747470733a2f2f6d6174706c6f746c69622e6f72672f5f7374617469632f6c6f676f322e737667" width="150" title="NumPY">
  <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/a/a1/PyCharm_Logo.svg/1024px-PyCharm_Logo.svg.png" width="50" title="Pycharm">
</p>

Python projects that showcase the Machine Learning algorithms: 

* Linear Regression with:
  * 1 variable
  * multiple variables
* Regularized Linear Regression
* Logistic Regression:
  * Linear
  * Non Linear
* Neural Networks:
  * using backpropagation on XOR
  * using Iris

<i>Linear Regression using 1 variable:</i>

Modifying the learning rate significantly effects how close the predicted output is to the training Y.
A learning rate too high (e.g. 1.0) made the algorithm overshoot and undershoot the target continuously and therefore gave a high Mean Square Error (i.e. making it very inaccurate). 
However, setting it to 0.0 meant the algorithm didn’t learn at all. 
I found that setting it to 0.01 gives the perfect balance between time to find optimal weights and accuracy of the results.
With a learning rate (alpha) of 0.01:

<p align="left">
  <img src="https://github.com/MalikS789/MachineLearning/blob/master/assgn_1_part_1/1_one_variable/figures/cost.png?raw=true" width="300">
  <img src="https://github.com/MalikS789/MachineLearning/blob/master/assgn_1_part_1/1_one_variable/figures/predictions.png?raw=true" width="340">
</p>

<i>Linear Regression using multiple variables:</i>

I Found that setting the learning rate to 0.01 gave the best reliability. 

<p align="left">
  <img src="https://github.com/MalikS789/MachineLearning/blob/master/assgn_1_part_1/2_multiple_variables/figures/cost.png?raw=true" width="300">
  <img src="https://github.com/MalikS789/MachineLearning/blob/master/assgn_1_part_1/2_multiple_variables/figures/predictions.png" width="300">
</p>

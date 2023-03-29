# Milestone 1

Dubien Victor: 282 279
Felber Arnaud: 302 283
Preto Anne-Valérie: 315 748

AICrowd team name : SAT
ID best submission : 142 020

## Dependencies
At first we had to be sure that our system supported : 
- pandas, 
- numpy, 
- matplotlib,
- torch. 
This is the case if you use deepnote. 
However, you need to add:

- `metrics.py`: Metrics to keep track of the loss and accuracy during training
- `helpers.py`: 
- `ipywidgets`: 


## Project structure

### Data
The data file had already been divided into three parts. 
This allows us to :
 - `train_set.csv` : to build a linear regression using the train inputs and outputs, 
 - `val_set.csv` : to verify our regressionon the validation (in order to avoid overfit)
 - `test_set.csv` : and to submit a CSV file of our outputs for the final test data file

### Code

The notebook `Milestone1.ipynb` was implemented to use a three-layer neural network and/or a HistGradientBoosting.
After experimenting with different regression techniques, we decided to work on the neural network.
We started with a simple model; the three-layer net shown during the exercise sessions. 
At first, we preprocess the data. We realised the most interesting part was to take the logarithm of our features. 
Indeed, the spectral acceleration varied very little and was between 0 and 1. 

We decided, given the speed of the test, not to necessarily correlate our base data for the neural network. 
We therefore left it as is. (We used the correlation for the GradientBoosting because we realized some of the 110 initial parameters were 
probably correlated).
For the use of the batch size and the shuffle, we had to reconsider how to define our inputs and outputs and to check the sizes.
Contrary to the example in the course, we are doing a linear regression model.
We therefore had to redefine our criteria to be able to observe the loss instead of the accuracy, which is used for classification.

We added batchnorms to our Neural Network, to renormalize the data over the batches, since the first normalization was over the whole 
training set.
We also added a dropout function to prevent overfitting (it drops a certain amount of parameters). 

It also seemed interesting to us to vary the learning rate according to the epoch. 
This is why we used a scheduler. Every 30 or 40 epochs (step_size), 
it makes our learning rate smaller by multiplying it by gamma (=0.1 or 0.2)

### How to run the notebok 
The Milestone1 notebook can be run from start to finish as is. 
It will predict with the neural network a csv file.
It is also possible to use GradientBoosting (long) or HistGradientBoosting by activating  them.

For our best prediction on A.I. crowd, the parameters were: 
batch_size : 128
dropout : p=0.1
scheduler : StepLR (step_size=30, gamma= 0.2)
epochs = 180
(For the val, we had 0.1330 MSE)

# ST-tau Implementation

This repository contains code for the paper 
"[Uncertainty Estimation and Calibration with Finite-State Probabilistic 
RNNs](https://openreview.net/forum?id=9EKHN1jOlA)" 
(Wang, Lawrence & Niepert, ICLR 2021).

## Installation
All dependencies ared install if ``python setup.py install`` 
or ``python setup.py develop`` is run. 
Please ensure that `pip` and `setuptools` are uptodate by 
running `pip install --upgrade pip setuptools`.


## Information
[ST-tau](st_tau/st_tau.py#L158) is implemented as a keras layer, inhereting from 
tensorflow.python.keras.layers.recurrent.PeepholeLSTMCell. 

ST-tau can be instantiated via:
````python
from st_tau.st_tau import STTAU
sttau_cell = STTAUCell(hidden_dim,
                       centroids=centroids,
                       temperature=temperature,
                       hard_sample=hard_sample)
````

A simple model can be built via:
````python
class STTAU_Model(Model):
    """
    Simple model that instantiates an embeddings layer, 
    STTAU and a dense output layer.
    Arguments specific to STTAUCell:
    centroids: the number of states to use. st-tau pecfic.
    temperature: temperature for Gumbel-Softmax. st-tau pecfic.
    hard_sample: If `True`, hard sample from Gumbel-Softmax. st-tau pecfic.
    """
    def __init__(self,
                 input_dim: int = 100,
                 hidden_dim: int = 128,
                 centroids: int = 5,
                 temperature: int = 1.,
                 hard_sample: bool = False,
                 ) -> None:
        super().__init__()
        self.input_embedding = Embedding(input_dim, hidden_dim)
        sttau_cell = STTAUCell(hidden_dim,
                               centroids=centroids,
                               temperature=temperature,
                               hard_sample=hard_sample)
        self.sttau = RNN(sttau_cell)
        self.output_layer = Dense(1, activation='sigmoid')

    def call(self, inputs, **kwargs):
        embedded = self.input_embedding(inputs)
        encoder_output = self.sttau(embedded)
        output = self.output_layer(encoder_output)
        return output
````

Given some input data, e.g.
````python
# load some data, here two test instances of IMDB with vocab size 100
input_data = np.array(
             [[0., 1., 1., 0., 1., 1., 1., 1., 1., 1., 1., 0., 0., 1., 1., 0.,
               1., 0., 0., 0., 0., 0., 0., 1., 0., 1., 0., 1., 1., 1., 0., 1.,
               1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 1., 0.,
               0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
               0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
               0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0.,
               0., 0., 0., 0.],
              [0., 1., 1., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
               0., 1., 0., 1., 1., 1., 1., 1., 0., 1., 0., 1., 1., 0., 1., 1.,
               0., 0., 1., 1., 0., 0., 1., 1., 0., 1., 0., 0., 0., 1., 0., 1.,
               1., 1., 1., 1., 1., 0., 0., 1., 1., 0., 1., 1., 1., 0., 0., 0.,
               0., 1., 1., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.,
               0., 0., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.,
               0., 0., 0., 0.]])
````

and an insantiated model,
````python
# instantiate model
tf.random.set_seed(42)  # set seed for reproducibility
sttau_model = STTAU_Model()
# train model here
````

model predictions and uncertainties can be obtained via:
````python
# run model predictions and compute uncertainty of the predictions
collect_predictions = []

# run prediction 10 times and collect output
for _ in range(10):
    output = sttau_model(input_data)
    collect_predictions.append(output.numpy())
collect_predictions = np.array(collect_predictions).squeeze()

# reshape so each row is the 10 predictions of one input
collect_predictions = np.transpose(collect_predictions)

# measure uncertainty (mean, variance and standard deviation)
mean = collect_predictions.mean(axis=1)
var = collect_predictions.var(axis=1)
std = collect_predictions.std(axis=1)

# print values
print('All predictions for each prediction: %s' % collect_predictions)
print('Mean for each prediction: %s' % mean)
print('Variance for each prediction: %s' % var)
print('Standard deviation for each prediction: %s' % std)

# asserts
np.testing.assert_almost_equal([0.50411403, 0.5054396], mean)
np.testing.assert_almost_equal([9.5507985e-06, 3.8244061e-06], var)
np.testing.assert_almost_equal([0.00309044, 0.00195561], std)
````

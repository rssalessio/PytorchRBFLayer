# Pytorch RBF Layer - Radial Basis Function Layer
Pytorch RBF Layer implements a radial basis function layer in Pytorch.

Radial Basis networks can be used to approximate functions, and can be combined together with other PyTorch layers.

An RBF is defined by 5 elements:
1. A radial kernel <img src="https://render.githubusercontent.com/render/math?math=\phi: [0,\infty) \to \mathbb{R}">
2. A positive shape parameter <img src="https://render.githubusercontent.com/render/math?math=\epsilon"> that is a scaling factor
3. The number of kernels <img src="https://render.githubusercontent.com/render/math?math=N">, and relative
   centers <img src="https://render.githubusercontent.com/render/math?math=\{c_i\}_{i=1}^N">
4. A norm <img src="https://render.githubusercontent.com/render/math?math=\|\cdot\|">
5. A set of weights <img src="https://render.githubusercontent.com/render/math?math=\{w_i\}_{i=1}^N">

The output of an RBF is given by
<img src="https://render.githubusercontent.com/render/math?math=y(x) = \sum_{i=1}^N w_i * \phi(\epsilon_i * ||x - c_i||)">, where <img src="https://render.githubusercontent.com/render/math?math=x"> is the input data.

The RBFLayer class takes as input: (1) the dimensionality of <img src="https://render.githubusercontent.com/render/math?math=x">; (2) the number of desired kernels; (3) the output dimensionality; (4) the radial function; (5) the norm to use.

The parameters can be either learnt, or set to a default parameter.    

For more information check 

* [1] https://en.wikipedia.org/wiki/Radial_basis_function
* [2] https://en.wikipedia.org/wiki/Radial_basis_function_network

_Author_: Alessio Russo (PhD Student at KTH - alessior@kth.se)


![alt tag](https://github.com/rssalessio/PytorchRBFLayer/blob/main/examples/img.png)

## License
Our code is released under the MIT license (refer to the [LICENSE](https://github.com/rssalessio/PytorchRBFLayer/blob/main/LICENSE) file for details).

## Requirements
To run the library you need atleast Python 3.5 and PyTorch.


## Usage/Examples
You can start using the layer by typing ```python from rbf_layer import RBFLayer``` in your code.

To learn how to use the RBFLayer, check the examples located in the examples/ folder.

In general the code has the following structure
```python
import torch
from rbf_layer import RBFLayer

# Define an RBF layer where the dimensionality of the input feature is 20,
# the number of kernels is 5, and 2 output features


# \ell norm
def l_norm(x, p=2):
    return torch.norm(x, p=p, dim=-1)


# Gaussian RBF
def rbf_gaussian(x):
    return (-x.pow(2)).exp()


# Use a radial basis function with euclidean norm
rbf = RBFLayer(in_features_dim=20,            # input features dimensionality
               num_kernels=5,                 # number of kernels
               out_features_dim=2,            # output features dimensionality
               radial_function=rbf_gaussian,  # radial basis function used
               norm_function=l_norm)          # l_norm defines the \ell norm


# Uniformly sample 100 points with 20 features
x = torch.rand((100, 20))

# Compute the output of the RBF layer
# y has shape(100, 2)
y = rbf(x)
```



## Citations
If you find this code useful in your research, please, consider citing it:
>@misc{pythonvrft,
>  author       = {Alessio Russo},
>  title        = {Pytorch RBF Layer},
>  year         = 2021,
>  doi          = {},
>  url          = { https://github.com/rssalessio/PytorchRBFLayer }
>}

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


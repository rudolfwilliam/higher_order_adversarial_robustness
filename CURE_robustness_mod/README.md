# CURE for adversarial vulnerability
CURE is a deep network training algorithm via a curvature regularizer. Networks trained using CURE can achieve significant adversarial robustness.

### Dependencies
---
The code is compatible with python3.7. To install the dependecies use
```
pip install -e .
```

To start, you can play with the following notebook:

* "[CURE Example Code](https://github.com/F-Salehi/CURE_robustness/blob/master/notebooks/example.ipynb) "


### Cuda
To install dependencies with cuda use
```
pip install -r requirements_c.txt -f https://download.pytorch.org/whl/cu113/torch_stable.html
```
Test that cuda is available with
```
python -c "import torch; print( torch.device('cuda' if torch.cuda.is_available() else 'cpu') )"
```

### Reference 
----
"[Robustness via curvature regularization, and vice versa](https://arxiv.org/abs/1811.09716) ", SM. Moosavi-Dezfooli, A. Fawzi, J. Uesato, and P. Frossard, _CVPR 2019_.

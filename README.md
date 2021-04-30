# @ML: A Toolbox for Adaptive Testing on Machine Learning Experiments

Introduction
------------

To develop a novel machine learning approach, we often encounter a scenario illustrated by the following table.


|             | Method 1    | Method 2   | Method 3   | ...        | New Method |
| ----------- | ----------- | ---------- | ---------- | ---------- | ---------- |
| Dataset 1   | xxx         | xxx        | xxx        | ...        | ???        |
| Dataset 2   | xxx         | xxx        | xxx        | ...        | ???        |
| Dataset 3   | xxx         | xxx        | xxx        | ...        | ???        |
| ...         | ...         | ...        | ...        | ...        | ...        |

While machine learning experiments used to be fast and cheap, some recent approaches might even take days to be tested with a single dataset.
And it becomes pretty difficult for the researchers to foreseen the overall performances on the entire selection of datasets.

This software package aims to tackle this common issue with Item Response Theory (IRT) and Adaptive Testing (AT). 
It allows the researchers to train different IRT models on the known results (e.g. from conference papers) and perform adaptive testing on any new approach under development.
Given the known experiment results, the algorithm can automatically select a sequence of datasets to learn the goodness of the new approach as soon as possible.
After each testing step, the algorithm can make performance inferences on the remaining datasets that haven't been tested yet.
These inferences allow the researchers to estimate the final results without spending prolonged running time.


User installation
-----------------

The @ML (at-ml) package can be installed from Pypi with the command

```
pip install at-ml
```

Documentation
-------------

The documentation can be found at https://at-ml.github.io/at-ml/

Example notebooks
-----------------

Servel example notebooks can be seen at https://github.com/AT-ML/at-ml/tree/main/notebooks

Citation
--------

If you use @ML in a scientific publication, it would be great if you can cite the following paper:

```
@article{song2021efficient,
  title={Efficient and Robust Model Benchmarks with Item Response Theory and Adaptive Testing.},
  author={Song, Hao and Flach, Peter},
  journal={International Journal of Interactive Multimedia \& Artificial Intelligence},
  volume={6},
  number={5},
  year={2021}
}
```

Acknowledgements
----------------

This toolbox is delivered as a part of the project: Measurement theory for data science and AI, funded by the Alan Turing Insititute (https://www.turing.ac.uk/). 

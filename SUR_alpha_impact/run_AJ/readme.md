
# Results of a posteriori estimates using the approximation of the Jeffreys prior or the constrained Jeffreys prior

## Description of the files

Each file `model_*` correspond to one run of a posetriori samples, they differ from their intial seed, so that they result from different batch of data.
1. Data are issued from the ASG piping system in non-linear behavior, and considering the PGA as IM. The data are collected from a dataset of 80000 simulations.
2. For each run, 250 data are collected.
3. For each run, there are 5000 a posteriori estimates of $\theta$ conditionaly to the first $k$ element of the given 250 data, for every $k\in\{10,20,\dots,250\}$.

The files `model_noSUR_J_*` correspond to the case where the prior was the Jeffreys prior, and the files `model_noSUR_J_adapt_*` correspond to the one where it was the constrained Jeffreys prior: $\pi\propto J\cdot \beta^{0.3}$.

The file are organized as follows:
1. They can be load uising pickle 
2. They contain a dictonary :
```{python}
dict = {
    'logs': dict{'post': list(numpy.array(5000,2) , size 26) } #contains the a posteriori samples
    'A': numpy.array(250) #contains the IMs
    'S': numpy.array(250) #contains the failure=1/non-failure=0
}
```





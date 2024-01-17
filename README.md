
## LINC Experiments

```
cd exp_synthetic
pip install -r requirements.txt
```

## Reproducing Plots
**General Instructions:**
To show a Figure, use the `plot_fig.py` files.

This will read the respective results from the `plots_paper` folder, show a plot, and print tex files to `plots_output`.

In `plots_paper/tex`, you can find the tex files currently used in the paper. They include for each method F1, Precision, Recall, and confidence bands.

The file name is `tex_VARIABLE_METRIC.csv`, where VARIABLE is the variable considered while the remaining settings are kept fixed, and METRIC is F1, Precision, Recall.

#### Fig. 2a: Causal Discovery Beyond MECs 
``` 
python plot_fig2a.py
```


#### Fig. 2b: Gene Regulatory Networks with SERGIO 
``` 
python plot_fig2b.py
```

####  Fig. 3: Robustness
``` 
python plot_fig3.py
```


## Reproducing Results 

**General Instructions:** Specify the desired experimental settings in `settings.py`, the base settings shown in the paper are used by default. 
Specify the methods that should be run also in `settings.py`.

####  Fig. 2a: Causal Discovery Beyond MECs 
``` 
python run_mec.py experiment --mec
```
Results will be in `results/identifier.csv` and can be read using `plot_mec.py` (after editing the experimental settings used). 


####  Fig. 2b: Gene regulatory networks with SERGIO
After setting the data simulator to `BASE_sim_data= ['sergio']` in `settings.py`, run

``` 
python run_mec.py experiment --mec
```


#### Fig. 3: Robustness 
By setting the data simulator to one of `BASE_sim_data= ['scale-iv', 'hard-iv', 'gp' ]` in `settings.py`, 
you can consider different functional models other than the standard `BASE_sim_data= ['cdnod' ]`. Then run


``` 
python run_mec.py experiment --mec
```

#### Fig. 4: Cell Signalling Data by Sachs et al. 

``` 
python run_sachs.py
```


#### LINC and its Variants 
In `settings.py`, you can set`'rff': True`, to use the Random Fourier Feature approximation of LINC.

With  `'ILP': False`, you'll be using the exhaustive search for partitions (which might be very slow!)
With `'ILP': True,  'clus' : False` you'll be using the clustering version of LINC with an ILP, With `'ILP': True,  'clus' : True` you'll be using the hierarchical, agglomerative clustering instead.

Finally, to experiment with DAG search algorithms, you can also set `'known_mec' : False` to run a full DAG search by LINC, that is, without knowing the Markov Equivalence class. For this, a greedy tree search algorithm inspired by the GES algorithm will be used.
 


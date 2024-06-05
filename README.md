# Causal Dicovery with GPs
[![compat](https://github.com/srhmm/linc/actions/workflows/compat.yml/badge.svg)](https://github.com/srhmm/linc/actions/workflows/compat.yml)
[![test](https://github.com/srhmm/linc/actions/workflows/test.yml/badge.svg)](https://github.com/srhmm/linc/actions/workflows/test.yml)
![Static Badge](https://img.shields.io/badge/python-%3E%3D3.8-blue?logo=python&label=python)
[![license](https://img.shields.io/github/license/machine-teaching-group/checkmate.svg)](https://github.com/srhmm/coco/blob/main/LICENSE)
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>

Based on Learning Causal Models Under Independent Changes (LINC).
 
### Setup
 
```
conda create -n "gpcd"
conda activate gpcd
pip install -r requirements.txt  
```

### Demo 
Bivariate Case
```
from gpcd.scoring import ScoreType, GPType, GPParams, score_edge
from testing.gen import gen_bivariate_example

data, truths, params, options = gen_bivariate_example()
gp_hyperparams = GPParams(ScoreType.GP, GPType.EXACT, None, None, None)
cause, effect = (0, 1) if truths.dag[0][1] != 0 else (1, 0)
logger = None
verbosity = 0

causal = score_edge(data.data_c, gp_hyperparams, [cause], effect, logger, verbosity)
acausal = score_edge(
    data.data_c, gp_hyperparams, [effect], cause, logger, verbosity
)
print(
    f"Causal dir. {causal:.2} {'<' if causal < acausal else '>'} "
    f"anticausal dir. {acausal:.2f} "
)

# Causal dir. 108.73 < anticausal dir. 1140.42  
```
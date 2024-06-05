# Causal Dicovery with GPs
[![compat](https://github.com/srhmm/linc/actions/workflows/compat.yml/badge.svg)](https://github.com/srhmm/linc/actions/workflows/compat.yml)
[![test](https://github.com/srhmm/linc/actions/workflows/test.yml/badge.svg)](https://github.com/srhmm/linc/actions/workflows/test.yml)
![Static Badge](https://img.shields.io/badge/python-%3E%3D3.8-blue?logo=python&label=python)
[![license](https://img.shields.io/github/license/machine-teaching-group/checkmate.svg)](https://github.com/srhmm/coco/blob/main/LICENSE)
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>

Causal discovery from continuous and multi-contexts data using GPs. Based on "Learning Causal Models Under Independent Changes (LINC)".
 
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
    f"Causal dir. {causal:.2f} {'<' if causal < acausal else '>'} "
    f"anticausal dir. {acausal:.2f} "
)
```
```
Causal dir. 108.73 < anticausal dir. 1140.42 
```
Gene expression data simulated with SERGIO (Dibaeinia and Sinha, 2020)
```
data, truths, params, options = gen_sergio_data()
options.exp_type = ExpType.MEC
print(
    f"SERGIO: {params['C']} experimental conditions over {params['N']} genes\nKnockout interventions (latent): "
    f"{', '.join(['Gene ' + str(tgt[0]) + ' in condition ' + str(i) for i, tgt in enumerate(truths.targets)])} "
)

methods = [LincMethod(), LincQFFMethod(), LincRFFMethod()]

for mthd in methods:
    mthd.fit(data, truths, params, options)

    print(
        f"Result for {mthd.nm()}:\tf1: {mthd.metrics['f1']:.2f}, shd: {mthd.metrics['shd']:.2f}, "
        f"tp: {mthd.metrics['tp']}, fp: {mthd.metrics['fp']}, time: {mthd.fit_times[0]:.2f}"
    )
```
```
SERGIO: 5 experimental conditions over 5 genes
Knockout interventions (latent): Gene 1 in condition 0, Gene 3 in condition 1, Gene 1 in condition 2, Gene 0 in condition 3, Gene 0 in condition 4 
Result for Linc:	f1: 1.00, shd: 0.00, tp: 5, fp: 0, time: 155.27
Result for LincQFF:	f1: 0.80, shd: 2.00, tp: 4, fp: 1, time: 130.45
Result for LincRFF:	f1: 0.80, shd: 2.00, tp: 4, fp: 1, time: 162.13
```
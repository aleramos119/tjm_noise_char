#%%
import numpy as np
import cma



def cma_opt(f, x0,sigma0, popsize=4):

    es = cma.CMAEvolutionStrategy(x0, sigma0, {
        'popsize': popsize,
        'verb_disp': 0,
        'bounds': [0, 1],
        # 'CMA_dampfac': 2.0,
    })

    while not es.stop():
        solutions = es.ask()
        values = [f(x) for x in solutions]
        es.tell(solutions, values)
        es.disp()

    result = es.result

    return result.fbest, result.xbest


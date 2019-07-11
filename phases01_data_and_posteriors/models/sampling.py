"""
This module provides the sampling, given a model and step.
"""

import pymc3 as pm
import scipy as sp
from pymc3.backends.base import merge_traces
import theano
import time
from timeit import default_timer as timer
from functools import partial
import pandas as pd
from copy import deepcopy

from .utils import format_trace
from .diagnostics import get_diagnostics
from . import MAX_NUM_SAMPLES, NUM_INIT_STEPS, SOFT_MAX_TIME_IN_SECONDS, \
              HARD_MAX_TIME_IN_SECONDS, MIN_SAMPLES_CONSTANT, NUM_CHAINS, \
              NUM_SCALE1_ITERS, NUM_SCALE0_ITERS, NUM_TUNE_BATCH_STEPS


def update_tuning_state(trace, step):
    mean_accept = trace.get_sampler_stats('mean_tree_accept')[-100:].mean()
    target_accept = 0.8
    n = mean_accept * 100
    lower, upper = sp.stats.beta(n + 1, 100 - n + 1).interval(0.95)
    if lower < target_accept < upper:
        pm.sampling.stop_tuning(step)
        return False
    else:
        return True


def sample_model(model, step=None, num_samples=MAX_NUM_SAMPLES, advi=False,
                 n_chains=NUM_CHAINS, raw_trace=False, single_chain=True,
                 num_scale1_iters=NUM_SCALE1_ITERS,
                 num_scale0_iters=NUM_SCALE0_ITERS,
                 random_seed=None):
    """
    Sample parallel chains from constructed Bayesian model.
    Returns tuple of Multitrace and diagnostics object.
    """
    sample_chain_with_args = partial(
        sample_chain, step=step, num_samples=num_samples, advi=advi,
        num_scale1_iters=num_scale1_iters, num_scale0_iters=num_scale0_iters,
        random_seed=random_seed)

    diagnostics = None
    if not advi:
        if single_chain:
            trace = sample_chain_with_args(model)
            diagnostics = get_diagnostics(trace, model, single_chain=True)
        else:
            traces = []
            for i in range(n_chains):
                print('chain {} of {}'.format(i + 1, n_chains))
                traces.append(sample_chain_with_args(model, chain_i=i))

            # copy and rebuild traces list because merge_traces modifies
            # the first trace in the list
            trace0 = deepcopy(traces[0])
            trace = merge_traces(traces)
            traces = [trace0] + traces[1:]

            diagnostics = get_diagnostics(merge_truncated_traces(traces),
                                          model, single_chain=False)
    else:
        trace = sample_chain_with_args(model)
        diagnostics = get_diagnostics(trace, model, single_chain=True)

    if raw_trace:
        return trace, diagnostics
    else:
        return format_trace(trace, variables=model.unobserved_RVs, to_df=True), diagnostics


def sample_chain(model, chain_i=0, step=None, num_samples=MAX_NUM_SAMPLES,
                 advi=False, tune_batch=NUM_TUNE_BATCH_STEPS, discard_tuned_samples=True,
                 num_scale1_iters=NUM_SCALE1_ITERS,
                 num_scale0_iters=NUM_SCALE0_ITERS,
                 random_seed=None):
    """Sample single chain from constructed Bayesian model"""
    with model:
        print('All unobserved random variables:', [_.name for _ in model.unobserved_RVs])
        print('Free random variables:', [_.name for _ in model.free_RVs])
        print('Total free variables dimension:', model.ndim)
        dim = model.ndim
        if not advi:
            print('Assigning NUTS sampler... [{time.ctime()}]')
            if step is None:
                start_, step = pm.init_nuts(
                    init='advi', njobs=1, n_init=NUM_INIT_STEPS, progressbar=False,
                    random_seed=random_seed if random_seed is not None else -1)

            min_num_samples = get_min_samples_per_chain(dim, MIN_SAMPLES_CONSTANT, NUM_CHAINS)
            last_print = None
            still_tuning = True
            start = timer()
            # We generate as many samples as we need and break afterwards, hence the 1 million
            # sample count below. We stop step size tuning manually, which is why tune is
            # None below.
            for i, trace in enumerate(pm.iter_sample(
                    1000000, step, start=start_, tune=None, chain=chain_i)):
                elapsed = timer() - start
                is_tuned = trace.get_sampler_stats('tune')
                tuned_samples = sum(is_tuned)
                real_samples = len(is_tuned) - tuned_samples
                if discard_tuned_samples:
                    discard = tuned_samples
                else:
                    discard = 0

                def print_status():
                    print(
                        f'   completed sample {i+1} '
                        f'({tuned_samples} tuned, {real_samples}/{num_samples} real) '
                        f'after {elapsed//3600:.0f}h{elapsed%3600/60:02.0f}m [{time.ctime()}]')

                if (last_print is None) or (elapsed - last_print > 300):
                    print_status()
                    last_print = elapsed

                # Check if we should stop tuning
                if still_tuning and (i + 1) % tune_batch == 0:
                    still_tuning = update_tuning_state(trace, step)
                    if not still_tuning:
                        print(f'   stopped tuning after iteration {i+1}')
                    else:
                        print(f'   still tuning after iteration {i+1}')

                # Do we have enough samples?
                if discard_tuned_samples and real_samples >= num_samples:
                    print_status()
                    assert real_samples == num_samples
                    break
                elif not discard_tuned_samples and real_samples + tuned_samples >= num_samples:
                    print_status()
                    assert real_samples + tuned_samples == num_samples
                    break

                # Has time run out?
                if elapsed > SOFT_MAX_TIME_IN_SECONDS / NUM_CHAINS:
                    print(f'exceeded soft time limit... [{time.ctime()}]')
                    if i + 1 - discard >= min_num_samples:
                        print('collected enough samples; stopping')
                        break
                    else:
                        print('but only collected {} of {}; continuing...'
                              .format(i + 1 - discard, min_num_samples))
                        if elapsed > HARD_MAX_TIME_IN_SECONDS / NUM_CHAINS:
                            print('exceeded HARD time limit; STOPPING')
                            break
            return trace[discard:]
        else:   # ADVI for neural networks
            print(f'Assigning ADVI... [{time.ctime()}]')
            min_num_samples = get_min_samples_per_chain(dim, MIN_SAMPLES_CONSTANT, 1)
            scale = theano.shared(pm.floatX(1))
            vi = pm.ADVI(cost_part_grad_scale=scale, random_seed=random_seed)
            print(f'Fitting model with {num_scale1_iters} iterations... [{time.ctime()}]')
            pm.fit(n=num_scale1_iters, method=vi)
            print(f'Fitting model with {num_scale0_iters} iterations... [{time.ctime()}]')
            scale.set_value(0)
            approx = pm.fit(n=num_scale0_iters)
            print(f'Drawing {min_num_samples} samples... [{time.ctime()}]')
            trace = approx.sample(draws=min_num_samples)
            return trace


def get_min_samples_per_chain(dimension, min_samples_constant, n_chains):
    # return int(min_samples_constant * (dimension ** 2) / n_chains)
    return MAX_NUM_SAMPLES


def augment_with_diagnostics(trace_df, diagnostics):
    """Add diagnostics to trace DataFrame"""
    d1 = diagnostics['Gelman-Rubin']
    d2 = diagnostics['ESS']
    if d1.keys() != d2.keys():
        raise ValueError('Diagnositics keys are not the same {} != {}'
                         .format(d1.keys(), d2.keys()))
    d_concat = {k: [d1[k], d2[k]] for k in d1.keys()}
    diag_df = pd.DataFrame.from_dict(d_concat)
    diag_df = diag_df.set_index('diagnostic')
    df_concat = pd.concat([diag_df, trace_df])
    return df_concat


def merge_truncated_traces(traces):
    min_chain_length = min(map(len, traces))
    truncated_traces = list(map(lambda trace: trace[-min_chain_length:],
                                traces))
    return merge_traces(truncated_traces)

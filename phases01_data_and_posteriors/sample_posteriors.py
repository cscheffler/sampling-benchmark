"""
File that draws samples from the posteriors of all of the supported models,
conditioned on all of the specified datasts.
"""
import random
import time
from joblib import Parallel, delayed
from functools import partial
from traceback import print_exc

from models.regression import REGRESSION_MODEL_NAMES, sample_regression_model
from models.classification import CLASSIFICATION_MODEL_NAMES, sample_classification_model
from data.repo_local import get_downloaded_dataset_ids_by_task
from data.io import read_dataset_Xy, read_dataset_categorical, write_samples, \
                    is_samples_file, append_sample_diagnostic
from data.config import Preprocess
from data.preprocessing.format import to_ndarray

NUM_MODELS_PER_DATASET = 1
'''
NUM_CORES_PER_CPU = 6
NUM_CPUS = 1
NUM_CORES = NUM_CPUS * NUM_CORES_PER_CPU
NUM_JOBS = int(NUM_CORES / 2)
'''

# For each different task (e.g. regression, classification, etc.),
# run outer loop that loops over datasets and inner loop that loops over models.
def sample_and_save_posteriors(dids, task, random_seed=None, start=None, stop=None):
    if task == 'regression':
        model_names = REGRESSION_MODEL_NAMES
        sample_model = sample_regression_model
    elif task == 'classification':
        model_names = CLASSIFICATION_MODEL_NAMES
        sample_model = sample_classification_model
    else:
        raise ValueError('Invalid task: ' + task)

    '''
    if random_seed is not None:
        random.seed(random_seed)
    random.shuffle(dids)
    '''
    
    #process_dataset_task = partial(process_dataset, model_names=model_names,
    #                               sample_model=sample_model)
    #Parallel(n_jobs=NUM_JOBS)(map(delayed(process_dataset_task), enumerate(dids)))
    for i, dataset_id in enumerate(dids):
        if (start is not None) and (i < start):
            continue
        if (stop is not None) and (i >= stop):
            break
        print(f'Iteration {i+1}/{len(dids)}, dataset id: {dataset_id} [{time.ctime()}]')
        process_dataset((i, dataset_id), model_names=model_names, sample_model=sample_model, random_seed=random_seed)


def process_dataset(i_and_dataset_id, model_names, sample_model, random_seed=None):
    """
    Sample from NUM_MODELS_PER_DATASET random model posteriors for the
    specified dataset. This function is run in parallel for many
    different datasets.
    """
    i, dataset_id = i_and_dataset_id
    # Partition datasets based on preprocessing
    if random_seed is not None:
        preprocess = [Preprocess.STANDARDIZED, Preprocess.ROBUST, Preprocess.WHITENED]
        my_random = random.Random(random_seed + dataset_id)
        index = my_random.randrange(len(preprocess))
        preprocess = preprocess[index]
        sampling_random_seed = my_random.randrange(2147462579)  # theano raises an error if the random seed is at or above this value
    else:
        if i % 3 == 0:
            preprocess = Preprocess.STANDARDIZED
        elif i % 3 == 1:
            preprocess = Preprocess.ROBUST
        elif i % 3 == 2:
            preprocess = Preprocess.WHITENED
        sampling_random_seed = None
        
    # Using random.sample() below has the nice property that, if we increase
    # or decrease NUM_MODELS_PER_DATASET, the first elements of the resulting
    # sub-list of model_names will remain the same. So, increasing the number
    # of models means the first few don't have to be recalculated and
    # decreasing it means nothing has to be recalculated.
    data_loaded = False
    for model_name in my_random.sample(model_names, NUM_MODELS_PER_DATASET):
        name = f'{dataset_id}_{preprocess.value}_{model_name}'
        if is_samples_file(model_name, dataset_id):
            print(name + ' samples file already exists... skipping')
            continue
        if not data_loaded:
            # Suboptimal: this information could be moved
            # into the same file to make things slightly faster
            num_non_categorical = read_dataset_categorical(dataset_id).count(False)
            X, y = read_dataset_Xy(dataset_id, preprocess)
            X = to_ndarray(X)
            data_loaded = True
        print('Starting sampling ' + name)
        try:
            samples, diagnostics = sample_model(
                model_name, X, y, num_non_categorical=num_non_categorical, random_seed=sampling_random_seed)
            print('Finished sampling ' + name)
            if samples is not None:
                write_samples(samples, model_name, dataset_id, overwrite=False)
                if diagnostics is not None:
                    diagnostics['name'] = name
                    append_sample_diagnostic(diagnostics)
                else:
                    append_sample_diagnostic({'name': name, 'advi': True})
            else:
                print(name, 'exceeded hard time limit, so it was discarded')
        except Exception:
            print('Exception on {}:'.format(name))
            print_exc()
        print('----------------------------------------')


if __name__ == '__main__':
    import sys

    task = sys.argv[1]
    if len(sys.argv) >= 3:
        start = int(sys.argv[2]) - 1   # 1 to start from beginning
    else:
        start = None
    if len(sys.argv) >= 4:
        stop = int(sys.argv[3]) - 1
    else:
        stop = None

    if task == 'regression':
        regression_dids = get_downloaded_dataset_ids_by_task('Supervised Regression')
        sample_and_save_posteriors(regression_dids, 'regression', random_seed=227934, start=start, stop=stop)
    elif task == 'classification':
        classification_dids = get_downloaded_dataset_ids_by_task('Supervised Classification')
        sample_and_save_posteriors(classification_dids, 'classification', random_seed=419112, start=start, stop=stop)

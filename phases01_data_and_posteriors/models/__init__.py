NUM_INIT_STEPS = 5000
NUM_TUNE_BATCH_STEPS = 100
MAX_NUM_SAMPLES = 1000
NUM_CHAINS = 1
SOFT_MAX_TIME_IN_MINUTES = 120
SOFT_MAX_TIME_IN_SECONDS = SOFT_MAX_TIME_IN_MINUTES * 60
HARD_MAX_TIME_IN_HOURS = 2
HARD_MAX_TIME_IN_SECONDS = HARD_MAX_TIME_IN_HOURS * 60 * 60
MIN_SAMPLES_CONSTANT = 10 
MAX_DATA_DIMENSION = {
    'linear': 25,
    'pairwise': 7,
    'quadratic': 6,
    'other': 100
}
MAX_GP_N = 1000
MAX_N = 50000
NUM_SCALE1_ITERS = 20000
NUM_SCALE0_ITERS = 30000
NUM_SCALE1_ITERS = 20
NUM_SCALE0_ITERS = 30

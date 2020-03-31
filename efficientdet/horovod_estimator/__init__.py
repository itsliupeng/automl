try:
    import horovod.tensorflow as hvd
except ImportError:
    hvd = None

import glob
import os
from multiprocessing import Pool, cpu_count

import numpy as np
import tensorflow as tf
from yacs.config import CfgNode

from .estimator import HorovodEstimator
from .utis import hvd_info, hvd_info_rank0, hvd_try_init


def _count_per_file(filename):
    c = 0
    for _ in tf.python_io.tf_record_iterator(filename):
        c += 1

    return c


def get_record_num(filenames):
    pool = Pool(cpu_count())
    c_list = pool.map(_count_per_file, filenames)
    total_count = np.sum(np.array(c_list))
    return total_count


def get_filenames(data_dir: str, filename_regexp: str, show_result=True):
    filenames = glob.glob(os.path.join(data_dir, filename_regexp))
    if show_result:
        hvd_info_rank0('find {} files in {}, such as {}'.format(len(filenames), data_dir, filenames[0:5]))
    return filenames

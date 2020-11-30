import logging
import os
from math import ceil
from multiprocessing.pool import ThreadPool
from typing import Union, Sized, Iterator

from tqdm import tqdm

from synthesis_classifier.model import get_tokenizer
from synthesis_classifier.pclassifier import MultiprocessingClassifier, paragraphs2batch, batch2numpy

__all__ = ['perform_collection', 'make_batch']

__author__ = 'Haoyan Huo'
__email__ = 'haoyan.huo@lbl.gov'
__maintainer__ = 'Haoyan Huo'


def make_batch(it: Union[Sized, Iterator], batch_size: int):
    def _iter():
        batch = []
        for i in it:
            batch.append(i)
            if len(batch) >= batch_size:
                yield batch
                del batch[:]

        if batch:
            yield batch

    class Batch:
        def __len__(self):
            return int(ceil(len(it) / batch_size))

        def __iter__(self):
            return _iter()

    return Batch()


def perform_collection(db_writer_cls, batch_generator, job_script=None):
    tokenizer = get_tokenizer()

    with db_writer_cls() as db_writer_queue:
        with MultiprocessingClassifier(db_writer_queue) as worker_queues:
            job_submitted = False

            worker = ThreadPool(1)
            len_result = worker.apply_async(batch_generator.__len__)

            with tqdm(desc='Classifying paragraphs', unit='batch') as pbar:
                for i, items in enumerate(batch_generator):
                    if not job_submitted and job_script is not None:
                        os.system(f'module load esslurm; sbatch -d singleton {job_script}')
                        job_submitted = True

                    if len_result.ready():
                        try:
                            total = len_result.get()
                            pbar.total = total
                        except Exception as e:
                            logging.exception('Cannot get length of batches: %r', e)

                    meta_ids, paragraphs = zip(*items)
                    _, batch = paragraphs2batch(paragraphs, tokenizer)
                    batch_numpy = batch2numpy(batch)
                    worker_queues[i % len(worker_queues)].put((meta_ids, batch_numpy))
                    pbar.update(1)

            worker.close()

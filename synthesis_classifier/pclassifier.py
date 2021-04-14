import logging
from multiprocessing import get_context
from typing import List, Dict, Tuple, Union

import torch
from numpy import ndarray
from scipy.special import softmax
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from transformers import BertTokenizerFast, BertForSequenceClassification

from synthesis_classifier.model import labels, max_input, get_model
from synthesis_classifier.utils import torch_dev

__all__ = [
    'run_batch', 'paragraphs2batch', 'get_classification_scores',
    'batch2tensor', 'batch2numpy',
    'MultiprocessingClassifier',
]

__author__ = 'Haoyan Huo'
__email__ = 'haoyan.huo@lbl.gov'
__maintainer__ = 'Haoyan Huo'


def get_classification_scores(model_output: Tensor) -> List[Dict]:
    """
    Compute classifier scores of a batched model output.

    :param model_output: Outputs Tensor[Batch, 5] from a classifier model.
    :return: List of classifier scores in dict.
    """
    outputs = model_output.cpu().numpy()
    scores = softmax(outputs, axis=1)

    results = []
    for i, _scores in enumerate(scores):
        _scores = {name: value.item() for
                   name, value in zip(labels, _scores)}

        results.append(_scores)
    return results


@torch.no_grad()
def run_batch(batch_text: List[str],
              model: BertForSequenceClassification,
              tokenizer: BertTokenizerFast) -> List[Dict]:
    """
    Run model classifier for a list of paragraphs.

    :param batch_text: List of paragraph strings.
    :param model: The paragraph classifier.
    :param tokenizer: The BERT tokenizer.
    :return: Classification result represented using a dict.
    """
    all_tokenized, batch = paragraphs2batch(batch_text, tokenizer)

    model.eval()
    # Make sure they are on the right dev
    for key, value in batch.items():
        batch[key] = value.to(torch_dev())

    outputs = model(**batch, return_dict=True).logits

    scores = get_classification_scores(outputs)
    results = [{
        'text': batch_text[i],
        'tokens': all_tokenized[i],
        'scores': scores[i]
    } for i in range(len(batch_text))]

    return results


def paragraphs2batch(paragraphs: List[str], tokenizer: BertTokenizerFast) -> \
        Tuple[List[List[str]], Dict]:
    """
    Convert a list of paragraphs to a batch. This essentially does these things:

    1. Tokenize paragraphs.
    2. Pad all input_ids tensors, remove excessively long input_ids.
    3. Generate the correct attention_masks.

    :param paragraphs: List of paragraphs.
    :param tokenizer: The BERT tokenizer.
    :return: Tokenized paragraphs and the batch that can be used as model inputs.
    """
    all_tokenized = []
    input_ids = []
    attention_mask = []

    for p in paragraphs:
        tokenized = tokenizer.tokenize(p)
        all_tokenized.append(tokenized)
        one_hot = tokenizer.convert_tokens_to_ids(tokenized)

        input_ids.append(torch.tensor(one_hot, dtype=torch.long))
        attention_mask.append(torch.ones_like(input_ids[-1], dtype=torch.float))

    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    input_ids = input_ids[:, :max_input]

    attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0.)
    attention_mask = attention_mask[:, :max_input]

    return all_tokenized, {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
    }


def batch2numpy(batch: Union[Dict, Tensor]):
    """
    Convert a batch into numpy arrays (that are easier to pickle).
    """
    if isinstance(batch, dict):
        return {key: batch2numpy(value) for key, value in batch.items()}
    elif isinstance(batch, Tensor):
        return batch.detach().cpu().numpy()
    else:
        raise TypeError('Unknown batch type %s to numpy' % type(batch))


def batch2tensor(batch: Union[Dict, ndarray]):
    """
    Convert a batch represented by numpy arrays into Tensors (for model input).
    """
    if isinstance(batch, dict):
        return {key: batch2tensor(value) for key, value in batch.items()}
    elif isinstance(batch, ndarray):
        return torch.from_numpy(batch).to(torch_dev())
    else:
        raise TypeError('Unknown batch type to tensor %s' % type(batch))


def subprocess_classifier(queue, db_writer_queue, dev_id=0):
    """
    This is the worker classifier that runs in a subprocess.
    It actively retrieves tokenized and batched (to maximize GPU utilization)
    inputs from a queue and performs paragraph classification. Then, the results
    are sent into a database writer queue to be collected and saved in database.

    The process can be terminated by sending a None as the EOF symbol.

    :param queue: A queue form which (meta_ids, batch_numpy) will be yielded.
    :param db_writer_queue: A queue to which (meta_ids, list of class scores) will be sent.
    :param dev_id: Optional device id to be set in PyTorch.
    """
    torch.cuda.set_device(dev_id)
    model = get_model()

    logging.info('Device %d ready to classify text.', dev_id)
    while True:
        batch_items = queue.get()
        if batch_items is None:
            break

        meta_ids, batch_numpy = batch_items
        batch = batch2tensor(batch_numpy)

        with torch.no_grad():
            output = model(**batch, return_dict=True)
            hidden_states = output.hidden_states[0].detach().cpu().numpy()
            scores = get_classification_scores(output.logits)

        db_writer_queue.put((meta_ids, scores, hidden_states))


class MultiprocessingClassifier(object):
    """
    A classifier that spawns multiple processes to use up all GPU
    resources installed on a machine.
    """

    def __init__(self, db_writer_queue):
        """
        Create a new multiprocessing classifier. The queues of subprocess
        classifiers can be obtained by accessing self.queues or using the
        following "with" codes:

        with MultiprocessingClassifier(writer_queue) as queues:
            # do stuff and send batches to queues

        :param db_writer_queue: The database writer queue that will be used by
            subprocess workers to send classifier results to.
        """
        n_gpus = torch.cuda.device_count()
        assert n_gpus > 0, "Must have at least one GPU!"
        logging.info('Spawning %d processes.', n_gpus)

        self.mp_ctx = get_context('spawn')

        self.queues = [self.mp_ctx.Queue(maxsize=16) for _ in range(n_gpus)]
        self.subprocesses = [self.mp_ctx.Process(
            target=subprocess_classifier, args=(self.queues[i], db_writer_queue, i)
        ) for i in range(n_gpus)]
        [process.start() for process in self.subprocesses]

    def __enter__(self):
        return self.queues

    def send_eof(self):
        """
        Send EOF to subprocess workers.
        """
        for q in self.queues:
            q.put(None)

    def wait(self):
        """
        Wait for subprocess workers to finish.
        """
        for process in self.subprocesses:
            process.join()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.send_eof()
        self.wait()

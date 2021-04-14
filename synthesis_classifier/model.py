import os
import sys
import urllib.request

from transformers import (
    BertTokenizerFast, BertForSequenceClassification,
    AutoConfig, set_seed,
    AutoModelForSequenceClassification)

from synthesis_classifier.utils import torch_dev

__all__ = [
    'labels', 'max_input', 'classifier_version',
    'get_tokenizer', 'get_model'
]

__author__ = 'Haoyan Huo'
__email__ = 'haoyan.huo@lbl.gov'
__maintainer__ = 'Haoyan Huo'

seed = 42
max_input = 512
labels = [
    'solid_state_ceramic_synthesis',
    'sol_gel_ceramic_synthesis',
    'hydrothermal_ceramic_synthesis',
    'precipitation_ceramic_synthesis',
    'something_else',
]
classifier_version = 'bert_classifier_20200904'

set_seed(seed)

this_dir = os.path.dirname(os.path.realpath(__file__))


def get_tokenizer() -> BertTokenizerFast:
    """
    Get the trained BERT tokenizer.
    """
    try:
        return BertTokenizerFast.from_pretrained(
            os.path.join(this_dir, 'data/models/fine-tuned-paragraph-classifier'),
            do_lower_case=False
        )
    except OSError:
        raise OSError('Failed to load model. Did you download the models by '
                      '`python -m synthesis_classifier.model download`?')


def get_model() -> BertForSequenceClassification:
    """
    Get the paragraph classifier model.
    """
    # Configs: basic configs
    num_labels = 5
    output_mode = 'classification'

    id2label = {str(i): l for i, l in enumerate(labels)}
    label2id = {l: str(i) for i, l in enumerate(labels)}
    output_dir = os.path.join(this_dir, 'data/models/fine-tuned-paragraph-classifier')

    try:
        config = AutoConfig.from_pretrained(
            output_dir,
            output_hidden_states=True,
            id2label=id2label,
            label2id=label2id,
            num_labels=num_labels,
            finetuning_task=output_mode,
        )

        model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=output_dir,
            config=config,
        )
    except OSError:
        raise OSError('Failed to load model. Did you download the models by '
                      '`python -m synthesis_classifier.model download`?')
    model.to(torch_dev())

    return model


def download():
    dest_dir = os.path.join(
        os.path.dirname(__file__),
        'data', 'models', 'fine-tuned-paragraph-classifier')
    os.makedirs(dest_dir)

    base_url = 'https://cedergroup-share.s3-us-west-2.amazonaws.com/' \
               'public/SynthesisClassifier/fine-tuned-paragraph-classifier-20201108/'

    files = [
        'config.json',
        'pytorch_model.bin',
        'special_tokens_map.json',
        'tokenizer_config.json',
        'training_args.bin',
        'validation-result.txt',
        'vocab.txt']

    for fn in files:
        url = f'{base_url}{fn}'
        print('Downloading', url)
        urllib.request.urlretrieve(url, os.path.join(dest_dir, fn))


if __name__ == '__main__':
    if len(sys.argv) == 2 and sys.argv[1] == 'download':
        download()

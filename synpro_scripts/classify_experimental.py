import re

from synthesis_classifier.model import classifier_version
from synthesis_classifier.multiprocessing_classifier import perform_collection, make_batch
from synthesis_classifier.database.synpro import MetaCollectionIteratorByQuery, SynProDBWriter


def experimental_paragraphs():
    query = {
        'path': re.compile('experiment|experimental|preparation|prepare|synthesis|syntheses|material', re.IGNORECASE),
        classifier_version: {'$exists': False}
    }

    return MetaCollectionIteratorByQuery(query)


if __name__ == "__main__":
    batch_size = 16
    perform_collection(
        SynProDBWriter,
        make_batch(experimental_paragraphs(), batch_size),
        './job_reclassify.sh'
    )

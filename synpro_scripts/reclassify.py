from synthesis_classifier.model import classifier_version
from synthesis_classifier.multiprocessing_classifier import perform_collection, make_batch
from synthesis_classifier.database.synpro import MetaCollectionIteratorByQuery, SynProDBWriter


def already_classified():
    query = {
        classifier_version: {'$exists': True},
        'classification': {'$exists': False}
    }

    return MetaCollectionIteratorByQuery(query)


if __name__ == "__main__":
    batch_size = 16
    perform_collection(
        SynProDBWriter,
        make_batch(already_classified(), batch_size),
        './job_reclassify.sh'
    )

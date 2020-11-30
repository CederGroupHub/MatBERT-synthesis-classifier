import re

from synthesis_classifier.multiprocessing_classifier import perform_collection, make_batch
from synthesis_classifier.database.patents import PatentsDBWriter, PatentParagraphsByQuery


def example_paragraphs():
    query = {
        'path': re.compile(r'.*example.*', re.IGNORECASE),
    }

    return PatentParagraphsByQuery(query)


if __name__ == "__main__":
    batch_size = 16
    perform_collection(
        PatentsDBWriter,
        make_batch(example_paragraphs(), batch_size),
        './job_patents_examples.sh'
    )

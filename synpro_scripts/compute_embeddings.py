from multiprocessing import get_context
from multiprocessing.queues import Queue

import numpy

from synthesis_classifier.database.synpro import MetaCollectionIteratorByQuery, get_connection
from synthesis_classifier.multiprocessing_classifier import perform_collection, make_batch


def not_embedding_paragraphs():
    query = {
        'paragraph_embedding': {'$exists': False}
    }

    return MetaCollectionIteratorByQuery(query)


class SynProEmbeddingWriter(object):
    def __init__(self):
        self.mp_ctx = get_context('spawn')  # To be compatible with classifier workers

        self.db_writer_queue = self.mp_ctx.Queue(maxsize=512)
        self.process = self.mp_ctx.Process(target=embedding_writer, args=(self.db_writer_queue,))
        self.process.start()

    def __enter__(self):
        return self.db_writer_queue

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.db_writer_queue.put(None)
        self.process.join()


def embedding_writer(queue: Queue):
    meta = get_connection().Paragraphs_Meta

    while True:
        batch_result = queue.get()
        if batch_result is None:
            break
        meta_ids, _, hidden_states = batch_result[:3]
        hidden_states = hidden_states.astype(numpy.float32)

        for meta_id, hs in zip(meta_ids, hidden_states):
            print(meta_id, hs.shape)


if __name__ == "__main__":
    batch_size = 16
    perform_collection(
        SynProEmbeddingWriter,
        make_batch(not_embedding_paragraphs(), batch_size),
    )

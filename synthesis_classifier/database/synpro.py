import os
from multiprocessing import Queue, get_context

from pymongo import MongoClient

from synthesis_classifier.model import classifier_version

__all__ = [
    'MetaCollectionIteratorByQuery',
    'SynProDBWriter'
]


def get_connection():
    client = MongoClient('synthesisproject.lbl.gov')
    db = client.SynPro

    assert 'SYNPRO_USERNAME' in os.environ and 'SYNPRO_PASSWORD' in os.environ, \
        "Please set SYNPRO_USERNAME and SYNPRO_PASSWORD"
    db.authenticate(os.environ['SYNPRO_USERNAME'], os.environ['SYNPRO_PASSWORD'])

    return db


class MetaCollectionIteratorByQuery(object):
    def __init__(self, query):
        self.db = get_connection()
        self.meta = self.db.Paragraphs_Meta
        self.query = query

    def __iter__(self):
        cursor = self.meta.aggregate([
            {'$match': self.query},
            {'$lookup': {
                'from': 'Paragraphs', 'localField': 'paragraph_id',
                'foreignField': '_id', 'as': 'paragraph'}},
            {'$unwind': '$paragraph'},
        ])

        for item in cursor:
            paragraph = item['paragraph']['text']
            if paragraph is not None and paragraph.strip():
                yield item['_id'], item['paragraph']['text']

    def __len__(self):
        return self.meta.find(self.query).count()


class SynProDBWriter(object):
    def __init__(self):
        self.mp_ctx = get_context('spawn')  # To be compatible with classifier workers

        self.db_writer_queue = self.mp_ctx.Queue(maxsize=512)
        self.process = self.mp_ctx.Process(target=db_annotate_process, args=(self.db_writer_queue,))
        self.process.start()

    def __enter__(self):
        return self.db_writer_queue

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.db_writer_queue.put(None)
        self.process.join()


def db_annotate_process(queue: Queue):
    meta = get_connection().Paragraphs_Meta

    while True:
        batch_result = queue.get()
        if batch_result is None:
            break
        meta_ids, scores = batch_result

        for meta_id, score in zip(meta_ids, scores):
            best_score = [(x, y) for (x, y) in score.items() if y > 0.5]
            classification = best_score[0][0] if best_score else None
            confidence = best_score[0][1] if best_score else None

            meta.update_one(
                {'_id': meta_id},
                {'$set': {
                    classifier_version: score,
                    'classification': classification,
                    'confidence': confidence,
                    'classifier_version': classifier_version,
                }}
            )

import os
from multiprocessing import Queue, get_context

from pymongo import MongoClient, HASHED, DESCENDING

from synthesis_classifier.model import classifier_version

__all__ = [
    'PatentParagraphsByQuery',
    'PatentsDBWriter'
]


def get_connection():
    client = MongoClient('synthesisproject.lbl.gov')
    db = client.Patents

    assert 'SYNPRO_USERNAME' in os.environ and 'SYNPRO_PASSWORD' in os.environ, \
        "Please set SYNPRO_USERNAME and SYNPRO_PASSWORD"
    db.authenticate(os.environ['SYNPRO_USERNAME'], os.environ['SYNPRO_PASSWORD'])

    return db


class PatentParagraphsByQuery(object):
    def __init__(self, query):
        self.db = get_connection()
        self.paragraphs = self.db.patent_text_section
        self.meta = self.db.patent_text_section_meta
        self.query = query

    @property
    def aggregate_pipelines(self):
        return [
            {'$match': self.query},
            {'$lookup': {
                'from': 'patent_text_section_meta', 'localField': '_id', 'foreignField': 'paragraph_id', 'as': 'meta'}},
            {'$project': {
                '_id': '$_id',
                'path': '$path',
                'text': '$text',
                'meta': {
                    '$filter': {
                        'input': '$meta',
                        'as': 's_meta',
                        'cond': {'$eq': ['$$s_meta.classifier_version', classifier_version]}
                    }
                }
            }},
            {'$match': {'meta': {'$size': 0}}}
        ]

    def __iter__(self):
        cursor = self.paragraphs.aggregate(self.aggregate_pipelines)

        for item in cursor:
            paragraph = item['text']
            if paragraph is not None and paragraph.strip():
                yield item['_id'], item['text']

    def __len__(self):
        return next(self.paragraphs.aggregate(self.aggregate_pipelines + [
            {'$count': 'total'}]))['total']


class PatentsDBWriter(object):
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
    meta = get_connection().patent_text_section_meta
    meta.create_index([('classification', HASHED)])
    meta.create_index('paragraph_id')
    meta.create_index([('classifier_version', HASHED)])
    meta.create_index([('confidence', DESCENDING)])

    while True:
        batch_result = queue.get()
        if batch_result is None:
            break
        paragraph_ids, scores = batch_result

        for paragraph_id, score in zip(paragraph_ids, scores):
            best_score = [(x, y) for (x, y) in score.items() if y > 0.5]
            classification = best_score[0][0] if best_score else None
            confidence = best_score[0][1] if best_score else None

            meta.update_one(
                {'paragraph_id': paragraph_id},
                {'$set': {
                    classifier_version: score,
                    'classification': classification,
                    'confidence': confidence,
                    'classifier_version': classifier_version,
                }},
                upsert=True
            )

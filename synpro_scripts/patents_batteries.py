from synthesis_classifier.database.patents import PatentsDBWriter, get_connection
from synthesis_classifier.multiprocessing_classifier import perform_collection, make_batch


class PatentBatteryParagraphs(object):
    def __init__(self):
        self.db = get_connection()

    def __iter__(self):
        cursor = self.db.patent_section_battery.aggregate([
            {'$lookup': {
                'from': 'patent_text_section_battery_meta',
                'localField': 'paragraph_id',
                'foreignField': 'paragraph_id',
                'as': 'meta'}},
            {'$match': {'meta': {'$size': 0}}},
            {'$lookup': {
                'from': 'patent_text_section',
                'localField': 'paragraph_id',
                'foreignField': '_id',
                'as': 'p'}},
        ])

        for item in cursor:
            paragraph = item['p'][0]['text']
            if paragraph is not None and paragraph.strip():
                yield item['paragraph_id'], item['p'][0]['text']

    def __len__(self):
        return next(self.db.patent_section_battery.aggregate([
            {'$lookup': {
                'from': 'patent_text_section_battery_meta',
                'localField': 'paragraph_id',
                'foreignField': 'paragraph_id',
                'as': 'meta'}},
            {'$match': {'meta': {'$size': 0}}},
            {'$count': 'total'}
        ]))['total']


if __name__ == "__main__":
    batch_size = 16
    perform_collection(
        PatentsDBWriter(meta_col_name='patent_text_section_battery_meta'),
        make_batch(PatentBatteryParagraphs(), batch_size),
        './job_patents_batteries.sh'
    )

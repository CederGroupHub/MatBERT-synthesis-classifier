db.getCollection('Paragraphs').find({}, {'path': 1}).forEach(function(doc) {
    db.Paragraphs_Meta.update(
        {paragraph_id: doc._id},
        {'$set': {
            path: doc.path,
        }}
    );
})

db.getCollection('Paragraphs_Meta').find({bert_classifier_20200904: {$exists: true}}).forEach(function(doc) {
    var result = doc.bert_classifier_20200904;

    if(result){
        var cls = result ? Object.keys(result).filter(i => result[i] > 0.5) : [];
        db.Paragraphs_Meta.update(
            {_id: doc._id}, 
            {'$set': {
                classification: cls.length > 0 ? cls[0] : null,
                confidence: cls.length > 0 ? result[cls[0]] : null,
                classifier_version: 'bert_classifier_20200904'
            }}
        );
    }
})

db.getCollection('Paragraphs_Meta').find({'bert_classifier_20200803.something_else': {$gte: 0.9}}).forEach(function(doc) {
    db.Paragraphs_Meta.update(
        {_id: doc._id},
        {'$set': {
            classification: 'something_else',
            confidence: doc.bert_classifier_20200803.something_else,
            classifier_version: 'bert_classifier_20200904'
        }}
    );
})
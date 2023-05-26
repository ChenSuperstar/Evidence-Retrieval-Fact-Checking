import json
import random

train_path = '../data/train-claims.json'
dev_path = '../data/dev-claims.json'
evi_path = '../data/reduced_evidences.json'


with open(train_path, 'r', encoding='utf8') as train, \
        open(dev_path, 'r', encoding='utf-8') as dev, \
        open(evi_path, 'r', encoding='utf-8') as evi,\
        open('../data/train-claims-neg.json', 'w', encoding='utf-8') as train_out,\
        open('../data/dev-claims-neg.json', 'w', encoding='utf-8') as dev_out:

    train_data = json.load(train)
    dev_data = json.load(dev)
    evidence_set = json.load(evi)

    for claim in train_data:
        pos_evidence = train_data[claim]["evidences"]
        count = 0
        select_evidence = [evidence for evidence in evidence_set if evidence not in pos_evidence]
        neg_evidence = random.sample(select_evidence, len(pos_evidence))
        train_data[claim]["neg_evidences"] = neg_evidence

    for claim in dev_data:
        pos_evidence = dev_data[claim]["evidences"]
        count = 0
        select_evidence = [evidence for evidence in evidence_set if evidence not in pos_evidence]
        neg_evidence = random.sample(select_evidence, len(pos_evidence))
        dev_data[claim]["neg_evidences"] = neg_evidence

    json.dump(train_data,train_out)
    json.dump(dev_data, dev_out)







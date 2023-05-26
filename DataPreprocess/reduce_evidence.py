import json
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import operator
from statistics import mean
from collections import Counter
from config import SIMILARITY

with open('../data/train-claims.json', 'r', encoding='utf-8') as train_claims:
    train_claims_data = json.load(train_claims)

with open('../data/dev-claims.json', 'r', encoding='utf-8') as dev_claims:
    dev_claims_data = json.load(dev_claims)

with open('../data/evidence.json', 'r', encoding='utf-8') as evidences:
    evidences_data = json.load(evidences)

with open('../data/test-claims-unlabelled.json', 'r', encoding='utf-8') as test:
    test_claims_data = json.load(test)

# reduced_evidence_index = []
#
# for claim in train_claims_data.keys():
#     reduced_evidence_index += train_claims_data[claim]['evidences']

evidence_ids = list(evidences_data.keys())
evidence_texts = list(evidences_data.values())

train_claims_ids = list(train_claims_data.keys())
train_claims_texts = [v['claim_text'] for v in train_claims_data.values()]

dev_claims_ids = list(dev_claims_data.keys())
dev_claims_texts = [v['claim_text'] for v in dev_claims_data.values()]

test_claims_ids = list(test_claims_data.keys())
test_claims_texts = [v['claim_text'] for v in test_claims_data.values()]

evidence_tfidf_vec = TfidfVectorizer(stop_words='english')
evidence_tfidf_vec.fit(evidence_texts+train_claims_texts+dev_claims_texts+test_claims_texts)
evidence_embedding_list = evidence_tfidf_vec.transform(evidence_texts)

claim_tfidf_vec = TfidfVectorizer()
train_claims_embedding_list = claim_tfidf_vec.fit_transform(train_claims_texts)
# dev_claims_embedding_list = claim_tfidf_vec.transform(dev_claims_texts)
# test_claims_embedding_list = claim_tfidf_vec.transform(test_claims_texts)

train_claims_data_temp = train_claims_data.copy()
dev_claims_data_temp = dev_claims_data.copy()
test_claims_data_temp = test_claims_data.copy()

evidence_out = []
for train_claim_id, train_claim in train_claims_data_temp.items():
    train_claim_embedding = evidence_tfidf_vec.transform([train_claim['claim_text']])
    similarity_dic = {}
    similarity = cosine_similarity(train_claim_embedding, evidence_embedding_list)[0]
    similarity = [(idx, cos) for (idx, cos) in enumerate(similarity)]
    similarity = sorted(similarity, key=lambda x: x[1], reverse=True)
    for i in range(len(similarity)):
        if similarity[i][1] > SIMILARITY:
            idx = similarity[i][0]
            evidence_out.append(evidence_ids[idx])
        else:
            break

for dev_claim_id, dev_claim in dev_claims_data_temp.items():
    dev_claim_embedding = evidence_tfidf_vec.transform([dev_claim['claim_text']])
    similarity_dic = {}
    similarity = cosine_similarity(dev_claim_embedding, evidence_embedding_list)[0]
    similarity = [(idx, cos) for (idx, cos) in enumerate(similarity)]
    similarity = sorted(similarity, key=lambda x: x[1], reverse=True)
    for i in range(len(similarity)):
        if similarity[i][1] > SIMILARITY:
            idx = similarity[i][0]
            evidence_out.append(evidence_ids[idx])
        else:
            break

for test_claim_id, test_claim in test_claims_data_temp.items():
    test_claim_embedding = evidence_tfidf_vec.transform([test_claim['claim_text']])
    similarity_dic = {}
    similarity = cosine_similarity(test_claim_embedding, evidence_embedding_list)[0]
    similarity = [(idx, cos) for (idx, cos) in enumerate(similarity)]
    similarity = sorted(similarity, key=lambda x: x[1], reverse=True)
    for i in range(len(similarity)):
        if similarity[i][1] > SIMILARITY:
            idx = similarity[i][0]
            evidence_out.append(evidence_ids[idx])
        else:
            break


for claim in train_claims_data.keys():
    evidence_out += train_claims_data[claim]['evidences']

for claim in dev_claims_data.keys():
    evidence_out += dev_claims_data[claim]['evidences']

evidence_out = list(set(evidence_out))

reduced_evidence ={}
for evidence in evidence_out:
    reduced_evidence[evidence] = evidences_data[evidence]

with open('../data/reduced_evidences.json','w', encoding='utf-8'):
    json.dump(reduced_evidence)







import json
import random
from statistics import mean

train_path = '../data/train-claims.json'
dev_path = '../data/dev-claims.json'
evi_path = '../data/evidence.json'

with open('../data/train-claims.json', 'r', encoding='utf-8') as train_claims:
    train_claims_data = json.load(train_claims)

with open('../data/dev-claims.json', 'r', encoding='utf-8') as dev_claims:
    dev_claims_data = json.load(dev_claims)

with open('../data/evidence.json', 'r', encoding='utf-8') as evidences:
    evidences_data = json.load(evidences)

with open('../data/test-claims-unlabelled.json', 'r', encoding='utf-8') as test:
    test_claims_data = json.load(test)

claim_count = 0
claim_length = []
evidence_count = []
labels = []

nums_key_evi = len(evidences_data.keys())
evidence_length = []
evidence_length_count = {}

for idx, text in evidences_data.items():
    evidence_length.append(len(text))
    if len(text) not in evidence_length_count:
        evidence_length_count[len(text)] = 1
    else:
        evidence_length_count[len(text)] += 1

max_evidence_len = max(evidence_length)
min_evidence_len = min(evidence_length)
mean_evidence_len = mean(evidence_length)
topk_evidence_len = sorted(evidence_length_count.items(), key=lambda x:x[1], reverse=True)


print("nums_key_evi: " + str(nums_key_evi))
# print(evidences_data['evidence-0'])
print("max_evidence_len: " + str(max_evidence_len))
print("min_evidence_len: " + str(min_evidence_len))
print("mean_evidence_len: " + str(mean_evidence_len))
print(topk_evidence_len  )
print('--------------------------')

#-----------------------------------------
min_ = float('inf')
max_ = 0
len_count = {}
class_count = {}
for obj in train_claims_data:
    nums_evidences = len(train_claims_data[obj]['evidences'])
    if nums_evidences <= min_:
        min_ = nums_evidences
    if nums_evidences >= max_:
        max_ = nums_evidences
    if nums_evidences not in len_count:
        len_count[nums_evidences] = 1
    else:
        len_count[nums_evidences] += 1

    class_ = train_claims_data[obj]['claim_label']
    if class_ not in class_count:
        class_count[class_] = 1
    else:
        class_count[class_] += 1

sum_ = 0
sum_value = 0
for key, value in len_count.items():
    sum_ += key * value
    sum_value += value

mean_ = sum_ / sum_value

print("min_evidence_in_claims: " + str(min_))
print("max_evidence_in_claims: " + str(max_))
print("mean_evidence_in_claims: " + str(mean_))
print("len_count_in_claims: " + str(len_count))
print("class_count_in_claims: " + str(class_count))
print('--------------------------')


from torch.utils.data import Dataset
import json
import random
from config import  TRAIN_CLAIMS_PATH, EVI_PATH, DEV_CLAIMS_PATH, TEST_PATH, EVIDENCE_COUNTS

class EvidenceDataset(Dataset):
    def __init__(self, tokenizer, max_length=128):
        self.max_length = max_length
        self.tokenizer = tokenizer

        self.evidences = self.data_load(EVI_PATH)
        self.evidences_ids = list(self.evidences.keys())

    def __len__(self):
        return len(self.evidences_ids)

    def __getitem__(self, idx):
        evidences_id = self.evidences_ids[idx]
        evidence = self.evidences[evidences_id]
        return [evidences_id, evidence]

    def data_load(self,text):
        f = open(text, "r")
        data = json.load(f)
        f.close()
        return data

    def collate_fn(self, batch):
        evidences_ids = []
        evidences = []

        for evidences_id, evidence in batch:
            evidences_ids.append(evidences_id)
            evidences.append(evidence.lower())

        evidences_text = self.tokenizer(
            evidences,
            max_length=self.max_length,
            padding=True,
            return_tensors="pt",
            truncation=True,
        )

        batch_encoding = dict()
        batch_encoding["evidence_input_ids"] = evidences_text.input_ids
        batch_encoding["evidence_attention_mask"] = evidences_text.attention_mask
        batch_encoding["evidences_ids"] = evidences_ids
        return batch_encoding


class TrainDataset(Dataset):
    def __init__(self, tokenizer,max_length=128):
        self.max_length = max_length
        self.tokenizer = tokenizer

        self.claims = self.data_load(TRAIN_CLAIMS_PATH)
        self.evidences = self.data_load(EVI_PATH)

        self.claim_ids = list(self.claims.keys())
        self.evidence_ids = list(self.evidences.keys())

        self.evidence_counts = EVIDENCE_COUNTS


    def __len__(self):
        return len(self.claim_ids)

    def __getitem__(self, idx):

        claim = self.claims[self.claim_ids[idx]]
        claim_text = claim['claim_text'].lower()
        evidences = claim['evidences']
        neg_evidences = claim['neg_evidences']
        return [claim_text, evidences, neg_evidences]

    def collate_fn(self, batch):
        claim_texts = []
        evidences = []
        neg_evidences =[]
        labels = []

        for claim_text, evidence, neg_evidence in batch:
            claim_texts.append(claim_text)
            evidences.extend(evidence)
            neg_evidences.extend(neg_evidence)
            labels.append(len(evidence))

        evidences.extend([neg for neg in neg_evidences if neg not in evidences])

        evidence_count = len(evidences)
        if evidence_count >= self.evidence_counts:
            evidences = evidences[:self.evidence_counts]
        evidences_text = [self.evidences[evidence_id].lower() for evidence_id in evidences]

        while evidence_count < self.evidence_counts:
            evidence_id = random.choice(self.evidence_ids)
            while evidence_id in evidences:
                evidence_id = random.choice(self.evidence_ids)
            evidences.append(evidence_id)
            evidences_text.append(self.evidences[evidence_id].lower())
            evidence_count += 1

        claim_enoder = self.tokenizer(
            claim_texts,
            max_length=self.max_length,
            padding=True,
            return_tensors="pt",
            truncation=True
        )

        evidence_encoder = self.tokenizer(
            evidences_text,
            max_length=self.max_length,
            padding=True,
            return_tensors="pt",
            truncation=True
        )

        batch_encoding = dict()
        batch_encoding['claim_input_ids'] = claim_enoder.input_ids
        batch_encoding['claim_attention_mask'] = claim_enoder.attention_mask
        batch_encoding['evidence_input_ids'] = evidence_encoder.input_ids
        batch_encoding['evidence_attention_mask'] = evidence_encoder.attention_mask
        batch_encoding['labels'] = labels
        return batch_encoding

    def data_load(self,text):
        f = open(text, "r")
        data = json.load(f)
        f.close()
        return data


class ValidationDataset(Dataset):
    def __init__(self, mode, tokenizer, max_length=512):
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.mode = mode

        self.claims = self.data_load(self.mode)
        self.claim_ids = list(self.claims.keys())

    def __len__(self):
        return len(self.claim_ids)

    def __getitem__(self, idx):
        claim = self.claims[self.claim_ids[idx]]
        claim_text = claim['claim_text'].lower()
        return [claim_text, claim, self.claim_ids[idx]]

    def data_load(self, mode):
        if mode == "test":
            f = open(TEST_PATH, "r")
        else:
            f = open(DEV_CLAIMS_PATH, "r")
        dataset = json.load(f)
        f.close()
        return dataset

    def collate_fn(self, batch):
        claim_texts = []
        claims = []
        evidences = []
        claim_ids = []
        for claim_text, claim, claim_id in batch:
            claim_texts.append(claim_text)
            claims.append(claim)
            if self.mode != "test":
                evidences.append(claim["evidences"])
            claim_ids.append(claim_id)

        claim_encoder = self.tokenizer(
            claim_texts,
            max_length=self.max_length,
            padding=True,
            return_tensors="pt",
            truncation=True,
        )

        batch_encoding = dict()
        batch_encoding["claim_input_ids"] = claim_encoder.input_ids
        batch_encoding["claim_attention_mask"] = claim_encoder.attention_mask

        batch_encoding["claims"] = claims
        batch_encoding["claim_ids"] = claim_ids
        if self.mode != "test":
            batch_encoding["evidences"] = evidences

        return batch_encoding
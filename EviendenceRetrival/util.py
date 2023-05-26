
def to_cuda(batch):
    for n in batch.keys():
        if n in ["claim_input_ids", "evidence_input_ids", "claim_attention_mask", "evidence_attention_mask"]:
            batch[n] = batch[n].cuda()
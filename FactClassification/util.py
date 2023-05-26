def to_cuda(batch):
    for n in batch.keys():
        if n in ["input_ids", "attention_mask", "label"]:
            batch[n] = batch[n].cuda()
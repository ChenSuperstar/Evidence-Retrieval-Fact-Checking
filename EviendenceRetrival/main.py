import json
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import wandb
from tqdm import tqdm #进度条
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel
from datetime import datetime
import argparse

from config import er_setting
from data_loader import TrainDataset, EvidenceDataset, ValidationDataset
from util import to_cuda

wandb.init(project="myproject", name="er")

def get_evidence_embeddings(evidence_dataloader, evidence_model):
    evidence_model.eval()
    # get evidence embedding and normalise
    evidence_ids = []
    evidence_embeddings = []
    for batch in tqdm(evidence_dataloader):
        to_cuda(batch)
        evidence_last = evidence_model(input_ids=batch["evidence_input_ids"], attention_mask=batch["evidence_attention_mask"]).last_hidden_state

        evidence_embedding = evidence_last[:, 0, :].detach()
        evidence_embedding_cpu = torch.nn.functional.normalize(evidence_embedding, p=2, dim=1).cpu()
        del evidence_embedding, evidence_last

        evidence_embeddings.append(evidence_embedding_cpu)
        evidence_ids.extend(batch["evidences_ids"])
    evidence_embeddings = torch.cat(evidence_embeddings, dim=0).t()
    return evidence_embeddings, evidence_ids

def validate(validation_dataloader, evidence_embeddings, evidence_ids, claim_model):
    f = []
    for batch in tqdm(validation_dataloader):
        to_cuda(batch)
        claim = claim_model(input_ids=batch["claim_input_ids"], attention_mask=batch["claim_attention_mask"]).last_hidden_state
        claim_embedding = claim[:, 0, :]
        claim_embedding = torch.nn.functional.normalize(claim_embedding, p=2, dim=1).cpu()
        scores = torch.mm(claim_embedding, evidence_embeddings)

        # cos_sims = torch.mm(claim_embedding, evidence_embeddings.t())
        # scores = - torch.nn.functional.log_softmax(cos_sims / 0.05, dim=1)

        topk_ids = torch.topk(scores, k=args.retrieval_num, dim=1).indices.tolist()

        for idx, data in enumerate(batch["claims"]):
            evidence_correct = 0
            pred_evidences = [evidence_ids[i] for i in topk_ids[idx]]
            for evidence_id in batch["evidences"][idx]:
                if evidence_id in pred_evidences:
                    evidence_correct += 1
            if evidence_correct > 0:
                evidence_recall = float(evidence_correct) / len(batch["evidences"][idx])
                evidence_precision = float(evidence_correct) / len(pred_evidences)
                evidence_fscore = (2 * evidence_precision * evidence_recall) / (evidence_precision + evidence_recall)
            else:
                evidence_fscore = 0
            f.append(evidence_fscore)

        # print("----")
    fscore = np.mean(f)
    print("\n")
    print("Evidence Retrieval F-score: %.3f" % fscore)
    print("\n")
    claim_model.train()
    return fscore

def predict(args):
    # load data
    er_setting(args)

    tokenizer = AutoTokenizer.from_pretrained(args.model_type)
    test_set = ValidationDataset("test", tokenizer, args.max_length)
    evidence_set = EvidenceDataset(tokenizer, args.max_length)

    dataloader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=4, collate_fn=test_set.collate_fn)
    evidence_dataloader = DataLoader(evidence_set, batch_size=128, shuffle=False, num_workers=4, collate_fn=evidence_set.collate_fn)

    # build models
    claim_model = AutoModel.from_pretrained(args.model_type)
    evidence_model = AutoModel.from_pretrained(args.model_type)

    assert len(args.model_pt) > 0

    claim_model.load_state_dict(torch.load(os.path.join("./cache", args.model_pt, "claim_ckpt.bin")))
    evidence_model.load_state_dict(torch.load(os.path.join("./cache", args.model_pt, "evidence_ckpt.bin")))

    claim_model.cuda()
    evidence_model.cuda()
    claim_model.eval()
    evidence_model.eval()

    # get evidence embedding and normalise
    evidence_ids = []
    evidence_embeddings = []
    for batch in tqdm(evidence_dataloader):
        to_cuda(batch)
        evidence_last = evidence_model(input_ids=batch["evidence_input_ids"], attention_mask=batch["evidence_attention_mask"]).last_hidden_state

        evidence_embedding = evidence_last[:, 0, :].detach()
        evidence_embedding_cpu = torch.nn.functional.normalize(evidence_embedding, p=2, dim=1).cpu()
        del evidence_embedding, evidence_last
        evidence_embeddings.append(evidence_embedding_cpu)
        evidence_ids.extend(batch["evidences_ids"])
    evidence_embeddings = torch.cat(evidence_embeddings, dim=0).t()

    out_data = {}
    for batch in tqdm(dataloader):
        to_cuda(batch)
        claim_last = claim_model(input_ids=batch["claim_input_ids"], attention_mask=batch["claim_attention_mask"]).last_hidden_state
        claim_embedding = claim_last[:, 0, :]
        claim_embedding = torch.nn.functional.normalize(claim_embedding, p=2, dim=1).cpu()
        scores = torch.mm(claim_embedding, evidence_embeddings)

        # cos_sims = torch.mm(claim_embedding, evidence_embeddings.t())
        # scores = - torch.nn.functional.log_softmax(cos_sims / 0.05, dim=1)

        topk_ids = torch.topk(scores, k=args.retrieval_num, dim=1).indices.tolist()
        for idx, data in enumerate(batch["claims"]):
            data["evidences"] = [evidence_ids[i] for i in topk_ids[idx]]
            out_data[batch["claim_ids"][idx]] = data
    fout = open("../data/retrieval-test-claims.json", 'w')
    json.dump(out_data, fout)
    fout.close()

def run(args):
    er_setting(args)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_type)

    train_set = TrainDataset(tokenizer=tokenizer, max_length=args.max_length)
    valid_set = ValidationDataset(mode='dev', tokenizer=tokenizer, max_length=args.max_length)
    evide_set = EvidenceDataset(tokenizer=tokenizer, max_length=args.max_length)

    train_dataloader = DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=True, num_workers=4, collate_fn=train_set.collate_fn)
    valid_dataloader = DataLoader(dataset=valid_set, batch_size=args.batch_size, shuffle=True, num_workers=4, collate_fn=valid_set.collate_fn)
    evide_dataloader = DataLoader(dataset=evide_set, batch_size=128, shuffle=True, num_workers=4, collate_fn=evide_set.collate_fn)

    # build models
    claim_model = AutoModel.from_pretrained(args.model_type)
    evidence_model = AutoModel.from_pretrained(args.model_type)

    if len(args.model_pt) > 0:
        claim_model.load_state_dict(torch.load(os.path.join("./cache", args.model_pt, 'claim_ckpt.bin')))
        evidence_model.load_state_dict(torch.load(os.path.join("./cache", args.model_pt, 'evidence_ckpt.bin')))

    claim_model.cuda()
    evidence_model.cuda()
    claim_model.eval()
    evidence_model.eval()

    date = datetime.now().strftime("%y-%m-%d")
    save_dir = f"./cache/{date}"
    os.makedirs(save_dir, exist_ok=True)

    claim_optimizer = optim.AdamW(claim_model.parameters())
    evidence_optimizer = optim.AdamW(evidence_model.parameters())

    for param_group in claim_optimizer.param_groups:
        param_group['lr'] = args.max_lr
    for param_group in evidence_optimizer.param_groups:
        param_group['lr'] = args.max_lr

    # start training
    claim_optimizer.zero_grad()
    evidence_optimizer.zero_grad()

    step_cnt = 0
    all_step_cnt = 0
    avg_loss = 0
    maximum_f_score = 0

    print("\nEvaluate:\n")
    # f_score = validate(val_dataloader, evidence_embeddings, evidence_ids, query_model, evidence_model)
    evidence_embeddings, evidence_ids = get_evidence_embeddings(evide_dataloader, evidence_model)
    f_score = validate(valid_dataloader, evidence_embeddings, evidence_ids, claim_model)
    wandb.log({"f_score": f_score}, step=all_step_cnt)

    for epoch in range(args.epoch):
        epoch_step = 0

        for (i, batch) in enumerate(tqdm(train_dataloader)):
            to_cuda(batch)
            step_cnt += 1
            # forward pass

            claim_embeddings = claim_model(input_ids=batch["claim_input_ids"], attention_mask=batch["claim_attention_mask"]).last_hidden_state
            evidence_embeddings = evidence_model(input_ids=batch["evidence_input_ids"],
                                           attention_mask=batch["evidence_attention_mask"]).last_hidden_state

            claim_embeddings = claim_embeddings[:, 0, :]
            evidence_embeddings = evidence_embeddings[:, 0, :]

            claim_embeddings = torch.nn.functional.normalize(claim_embeddings, p=2, dim=1)
            evidence_embeddings = torch.nn.functional.normalize(evidence_embeddings, p=2, dim=1)

            cos_sims = torch.mm(claim_embeddings, evidence_embeddings.t())
            scores = - torch.nn.functional.log_softmax(cos_sims/0.05, dim=1)

            loss = []
            start_idx = 0
            for idx, label in enumerate(batch["labels"]):
                end_idx = start_idx + label
                cur_loss = torch.mean(scores[idx, start_idx:end_idx])
                loss.append(cur_loss)
                start_idx = end_idx

            loss = torch.stack(loss).mean()
            loss = loss / args.accumulate_step
            loss.backward()

            avg_loss += loss.item()
            if step_cnt == args.accumulate_step:
                # updating
                if args.grad_norm > 0:
                    nn.utils.clip_grad_norm_(claim_model.parameters(), args.grad_norm)
                    nn.utils.clip_grad_norm_(evidence_model.parameters(), args.grad_norm)

                step_cnt = 0
                epoch_step += 1
                all_step_cnt += 1

                # adjust learning rate
                if all_step_cnt <= args.warmup_steps:
                    lr = all_step_cnt * (args.max_lr - 2e-8) / args.warmup_steps + 2e-8
                else:
                    lr = args.max_lr - (all_step_cnt - args.warmup_steps) * 1e-8

                # keep lr fixed
                for param_group in claim_optimizer.param_groups:
                    param_group['lr'] = lr

                for param_group in evidence_optimizer.param_groups:
                    param_group['lr'] = lr

                claim_optimizer.step()
                evidence_optimizer.step()

                claim_optimizer.zero_grad()
                evidence_optimizer.zero_grad()

            if all_step_cnt % args.report_freq == 0 and step_cnt == 0:
                if all_step_cnt <= args.warmup_steps:
                    lr = all_step_cnt * (args.max_lr - 2e-8) / args.warmup_steps + 2e-8
                else:
                    lr = args.max_lr - (all_step_cnt - args.warmup_steps) * 1e-8

                wandb.log({"learning_rate": lr}, step=all_step_cnt)
                wandb.log({"loss": avg_loss / args.report_freq}, step=all_step_cnt)
                # report stats
                print("\n")
                print("epoch: %d, epoch_step: %d, avg loss: %.6f" % (epoch + 1, epoch_step, avg_loss / args.report_freq))
                print(f"learning rate: {lr:.6f}")
                print("\n")

                avg_loss = 0
            del loss, cos_sims, claim_embeddings, evidence_embeddings

            if all_step_cnt % args.eval_interval == 0 and all_step_cnt != 0 and step_cnt == 0:
                # evaluate the model as a scorer
                print("\nEvaluate:\n")
                # f_score = validate(val_dataloader, evidence_dataloader, query_model, evidence_model)
                evidence_embeddings, evidence_ids = get_evidence_embeddings(evide_dataloader, evidence_model)
                f_score = validate(valid_dataloader, evidence_embeddings, evidence_ids, claim_model)
                wandb.log({"f_score": f_score}, step=all_step_cnt)

                if f_score > maximum_f_score:
                    maximum_f_score = f_score
                    torch.save(claim_model.state_dict(), os.path.join(save_dir, "claim_ckpt.bin"))
                    torch.save(evidence_model.state_dict(), os.path.join(save_dir, "evidence_ckpt.bin"))
                    print("\n")
                    print("best val loss - epoch: %d, epoch_step: %d" % (epoch, epoch_step))
                    print("maximum_f_score", f_score)
                    print("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parameters') #参数解析工具
    parser.add_argument("-p", "--predict", action="store_true", help="predict test using the best model")
    parser.add_argument("--model_pt", default="", type=str, help="model path") #check point
    args = parser.parse_args()

    if args.predict:
        predict(args)
    else:
        run(args)
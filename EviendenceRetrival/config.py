TRAIN_CLAIMS_PATH = '../data/train-claims-neg.json'
DEV_CLAIMS_PATH = '../data/dev-claims-neg.json'
EVI_PATH = '../data/reduced-evidences.json'
TEST_PATH = '../data/test-claims-unlabelled.json'
EVIDENCE_COUNTS = 96

def er_setting(args):
    # default setting
    args.batch_size = getattr(args, 'batch_size', 24)
    args.epoch = getattr(args, 'epoch', 60)
    args.report_freq = getattr(args, "report_freq", 5)
    args.accumulate_step = getattr(args, "accumulate_step", 2)
    args.model_type = getattr(args, "model_type", "bert-based-uncased")
    args.warmup_steps = getattr(args, "warmup_steps", 50)
    args.grad_norm = getattr(args, "grad_norm", 1)
    args.seed = getattr(args, "seed", 42)
    args.max_lr = getattr(args, "max_lr", 2e-5)
    args.max_length = getattr(args, "max_length", 128)
    args.eval_interval = getattr(args, "eval_interval", 20) #评估次数
    args.retrieval_num = getattr(args, "retrieval_num", 4)
    args.evidence_samples = getattr(args, "evidence_samples", 64) #一个batch里面

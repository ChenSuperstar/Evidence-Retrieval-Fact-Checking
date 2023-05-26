TRAIN_CLAIMS_PATH = '../data/train-claims-neg.json'
DEV_CLAIMS_PATH = '../data/dev-claims-neg.json'
EVI_PATH = '../data/reduced-evidences.json'
TEST_PATH = '../data/test-claims-unlabelled.json'

def fc_setting(args):
    # default setting
    args.batch_size = getattr(args, 'batch_size', 12)
    args.epoch = getattr(args, 'epoch', 20)
    args.report_freq = getattr(args, "report_freq", 10)
    args.accumulate_step = getattr(args, "accumulate_step", 4)
    args.model_type = getattr(args, "model_type", "bert-based-uncased")
    args.warmup_steps = getattr(args, "warmup_steps", 200)
    args.grad_norm = getattr(args, "grad_norm", 1)
    args.seed = getattr(args, "seed",42)
    args.max_lr = getattr(args, "max_lr", 1e-5)
    args.max_length = getattr(args, "max_length", 512)
    args.eval_interval = getattr(args, "eval_interval", 20)

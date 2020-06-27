from processor import Others_OneSentence_Processor
from Trainer import Trainer
import os
model_dir = 'result/'
class Args(object):
    pass
args = Args()

args.bert_model = 'bert-base-uncased'
args.do_lower_case = True
args.warmup_proportion = 0.1
args.cache_dir = "./cache"
args.no_cuda = False
args.local_rank = -1
args.fp16 = False
args.loss_scale = 0
args.gradient_accumulation_steps = 1
args.server_ip = ''
args.server_port = ''
args.output_mode = "classification"
args.save_model_steps = 2000
args.resume_epochs = 0
args.resume_steps = 0

# Important configurations
args.data_dir = 'dataset/preprocessed/'
args.train_file = 'train.json'
args.dev_file = 'dev.json'
args.train_batch_size = 32
args.eval_batch_size = 32
args.do_train = True
args.do_eval = False
args.do_run = False
args.num_train_epochs = 8.0
args.max_seq_length = 256
args.processor = Others_OneSentence_Processor
args.output_dir = os.path.join(model_dir, 'friends_others')
args.resume_dir = None

args.learning_rate = 1e-5
args.seed = 1991

trainer = Trainer(args)
trainer.execute()


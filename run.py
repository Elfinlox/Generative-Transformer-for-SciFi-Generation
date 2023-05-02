import torch
import torch.nn as nn
import re
import pickle
import itertools
from torch.nn import functional as F

from model import *
from train import *
from prepare import *

import argparse 

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', help='Name of the model')
parser.add_argument('-c', '--context', help = 'Context passed to model')

args = parser.parse_args()

model = torch.load(f"Model/{args.model}").to(device)
model.eval()

context = r"%s" % args.context
context = torch.tensor(tokenizer.encode(context).ids, dtype = torch.long, device = device).view(1, -1)
print(tokenizer.decode(model.generate(config, context, eos = None, max_len=100)[0].tolist()))

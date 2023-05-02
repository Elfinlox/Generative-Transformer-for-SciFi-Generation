import os
import pickle
import re
import numpy as np

from tokgen import *

OUTPUT_FOLDER = "Data/"

if not os.path.exists(OUTPUT_FOLDER):
    os.mkdir(OUTPUT_FOLDER)

tokenizer = Tokenizer.from_file("Data/tokenizer.json")

if __name__ == "__main__":
    text = tokenizer.encode(open("Data/scifi.txt").read())

    vocab = tokenizer.get_vocab()
    vocab_size = tokenizer.get_vocab_size()

    print("Size of dataset: ", len(text))
    print("Vocab Size: ", vocab_size)

    n = len(text)
    train_data = np.array(text.ids[:int(n*0.9)], dtype = np.uint16)
    val_data = np.array(text.ids[int(n*0.9):], dtype = np.uint16)

    train_data.tofile(OUTPUT_FOLDER + "train.bin")
    val_data.tofile(OUTPUT_FOLDER + "val.bin")

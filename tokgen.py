from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

if __name__ == "__main__":
    tokenizer = Tokenizer(BPE(unk_token = "[UNK]"))
    trainer = BpeTrainer(vocab_size = 10000, min_frequency = 2, special_tokens = ["[UNK]"])
    tokenizer.pre_tokenizer = Whitespace()

    files = ["Data/scifi.txt"]

    tokenizer.train(files, trainer)
    tokenizer.save("Data/tokenizer.json")
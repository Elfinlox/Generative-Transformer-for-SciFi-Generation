# Generative-Transformer-for-SciFi-Generation
A GPT like model using PyTorch to generate Sci-Fi

1) Download the Dataset from ```https://www.kaggle.com/datasets/jannesklaas/scifi-stories-text-corpus```.
2) Run ```tokgen.py``` to generate the tokenizer ```tokenizer.json```.
3) Run ```prepare.py``` to read the dataset and split it into train and validation sets and store it as memory maps.
4) Run ```train.py``` to train the model. Edit parameters inside the file as required.
5) Use ```python run.py -m [Model-name] -c [Text-prompt]``` to complete your prompts using Sci-Fi.

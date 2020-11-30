import torch

from synthesis_classifier import get_model, get_tokenizer, run_batch

model = get_model()
tokenizer = get_tokenizer()

with open('examples.txt', 'r') as f:
    paragraphs = list(map(str.strip, f))

batch_size = 2
batches = [paragraphs[i:min(i + batch_size, len(paragraphs))]
           for i in range(0, len(paragraphs), batch_size)]

for batch in batches:
    result = run_batch(batch, model, tokenizer)
    print(result)

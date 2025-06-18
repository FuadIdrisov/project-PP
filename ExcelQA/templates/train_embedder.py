from sentence_transformers import SentenceTransformer, losses, InputExample
from torch.utils.data import DataLoader
import json

model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

train_examples = []
with open("train_data.jsonl", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        train_examples.append(InputExample(texts=obj["texts"]))

train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
train_loss = losses.MultipleNegativesRankingLoss(model)

model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=20,
    warmup_steps=10,
    output_path="trained_embed_model"
)

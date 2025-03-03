import numpy as np
import pickle
from nltk import tokenize
import torch
from torch import optim, nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from data_processing import load_data_parallel

class Word2VecDataset(Dataset):
    """Dataset for training Word2Vec model on congressional speeches with CUDA support."""
    def __init__(self, texts_with_metadata, word_to_index, window_size=5, num_negatives=5):
        self.word_to_index = word_to_index
        self.vocab_size = len(word_to_index)
        self.window_size = window_size
        self.num_negatives = num_negatives
        self.document_labels = []
        self.pairs = []

        # Process all documents 📄✨
        for doc_idx, (sentences, label, speaker) in enumerate(texts_with_metadata):
            self.document_labels.append((label, speaker))
            for sentence in sentences:
                word_indices = [word_to_index.get(word, word_to_index['<UNK>'])
                                for word in sentence]
                for i, center in enumerate(word_indices):
                    start = max(0, i - window_size)
                    end = min(len(word_indices), i + window_size + 1)
                    context = [j for j in range(start, end) if j != i]
                    for j in context:
                        self.pairs.append((center, word_indices[j], doc_idx))

        # Convert pairs to tensors for faster GPU access
        self.pairs = torch.tensor(self.pairs, dtype=torch.long)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        center_idx, context_idx, doc_idx = self.pairs[idx]
        device = self.pairs.device
        negatives = torch.randint(0, self.vocab_size, (self.num_negatives,),
                                  device=device)
        mask = negatives == context_idx
        if mask.any():
            negatives[mask] = (negatives[mask] + 1) % self.vocab_size

        return (center_idx, context_idx, negatives, doc_idx)

    def get_document_labels(self):
        return self.document_labels


class Word2Vec(nn.Module):
    """
    Word2Vec Skipgram. I choose skipgram out of habit, and I wanted to train from scratch, as I want to
    compare if pre-trained word embeddings will put bias onto this data.
    """

    def __init__(self, vocab_size, embedding_dim):
        super(Word2Vec, self).__init__()
        self.input_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.output_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # Initialize weights with better scaling for GPU
        std = 1.0 / np.sqrt(embedding_dim)
        nn.init.normal_(self.input_embeddings.weight, mean=0.0, std=std)
        nn.init.normal_(self.output_embeddings.weight, mean=0.0, std=std)

    def forward(self, center_words, context_words, negative_words):
        # Compute embeddings
        center_embeds = self.input_embeddings(center_words)
        context_embeds = self.output_embeddings(context_words)
        negative_embeds = self.output_embeddings(negative_words)

        pos_scores = torch.sum(center_embeds * context_embeds, dim=1)
        pos_loss = F.logsigmoid(pos_scores)

        center_embeds = center_embeds.unsqueeze(1)
        neg_scores = torch.bmm(negative_embeds, center_embeds.transpose(1, 2)).squeeze()
        neg_loss = F.logsigmoid(-neg_scores).sum(dim=1)

        return -(pos_loss + neg_loss).mean()


def train_word2vec(dataset, vocab_size, embedding_dim=50, batch_size=512,
                   num_epochs=10, num_negatives=5, learning_rate=0.01):

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    # Set up device and model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"Training on GPU: {torch.cuda.get_device_name(0)}")

    model = Word2Vec(vocab_size, embedding_dim).to(device)
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    # AdamW for speed 💨
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=1,
        verbose=True
    )

    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")

        for batch_idx, (center, context, negatives, _) in enumerate(progress_bar):
            # GPUGPUGPUGPUGPU
            center = center.to(device)
            context = context.to(device)
            negatives = negatives.to(device)

            if scaler is not None:
                with torch.cuda.amp.autocast():
                    loss = model(center, context, negatives)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss = model(center, context, negatives)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

            optimizer.zero_grad()
            total_loss += loss.item()
            avg_loss = total_loss / (batch_idx + 1)
            progress_bar.set_postfix({'avg_loss': f'{avg_loss:.4f}'})

        scheduler.step(avg_loss)

    return model.input_embeddings.weight.data.cpu().numpy()

def run_training(): # let's gooooo
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")

    data_dir = "Data"
    texts, labels, speakers = load_data_parallel(data_dir, multilabel=False)
    print(f"Loaded {len(texts)} documents.")

    # Tokenize 🪓
    texts_with_metadata = []
    for text, label, speaker in zip(texts, labels, speakers):
        sentences = tokenize.sent_tokenize(text)
        tokenized_sentences = [sent.lower().split() for sent in sentences if sent.strip()]
        if tokenized_sentences:
            texts_with_metadata.append((tokenized_sentences, label, speaker))

    # build the vocab 📚
    vocab = {'<UNK>': 0}
    for sentences, _, _ in texts_with_metadata:
        for sent in sentences:
            for word in sent:
                if word not in vocab:
                    vocab[word] = len(vocab)

    word_to_index = vocab
    vocab_size = len(vocab)
    print(f"Vocabulary size (including <UNK>): {vocab_size}")

    # train word2vec 🧠
    dataset = Word2VecDataset(texts_with_metadata, word_to_index)
    w_in = train_word2vec(dataset,
                          vocab_size=vocab_size,
                          embedding_dim=50,
                          batch_size=1024,
                          num_epochs=5,
                          num_negatives=5,
                          learning_rate=0.001)

    # save for later alligator 🐊
    np.save("results_and_embeddings/w_in_torch.npy", w_in)
    with open("results_and_embeddings/word_to_index.pkl", "wb") as f:
        pickle.dump(word_to_index, f)
    with open("results_and_embeddings/document_labels.pkl", "wb") as f:
        pickle.dump(dataset.get_document_labels(), f)

if __name__ == '__main__':
    run_training()
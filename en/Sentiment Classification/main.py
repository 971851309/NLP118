import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.preprocessing import LabelEncoder
from torch.nn.utils.rnn import pad_sequence
import random


texts = [
    "I feel so sad today, everything is going wrong.",
    "I'm really angry about what happened!",
    "I'm anxious about the future, I don't know what to do.",
    "I'm so happy, today is a great day!",
    "I'm scared of what might happen next.",
    "I'm disappointed with the results, I worked so hard."
]
labels = ["sad", "angry", "anxious", "happy", "scared", "disappointed"]  # 情感标签


responses = {
    "sad": ["I'm sorry to hear that you're feeling sad. Things will get better.", "It's okay to feel down sometimes. You're not alone."],
    "angry": ["Take a deep breath and try to calm down. Everything will be okay.", "I understand you're upset. Let's work through this together."],
    "anxious": ["Try to relax. Things are not as bad as they seem.", "Take it one step at a time. You've got this!"],
    "happy": ["That's great to hear! Keep up the good mood!", "I'm so glad you're feeling happy!"],
    "scared": ["Don't be afraid. I'm here to support you.", "You're stronger than you think. You can face this!"],
    "disappointed": ["Disappointments are temporary. There are always new opportunities.", "Don't lose hope. Every setback is a step toward success."]
}


class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.label_encoder = LabelEncoder()
        self.labels = self.label_encoder.fit_transform(self.labels)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        tokens = self.tokenizer.texts_to_sequences([text])[0]  # 使用 texts_to_sequences 方法
        return torch.tensor(tokens, dtype=torch.long), torch.tensor(label, dtype=torch.long)


class SimpleTokenizer:
    def __init__(self):
        self.vocab = {}
        self.index = 1

    def fit_on_texts(self, texts):
        for text in texts:
            for word in text.split():
                if word not in self.vocab:
                    self.vocab[word] = self.index
                    self.index += 1

    def texts_to_sequences(self, texts):
        sequences = []
        for text in texts:
            sequence = [self.vocab.get(word, 0) for word in text.split()]  # 将单词映射为整数
            sequences.append(sequence)
        return sequences

#  LSTM模型
class EmotionClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, dropout):
        super(EmotionClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        embedded = self.dropout(self.embedding(text))
        output, (hidden, cell) = self.lstm(embedded)
        return self.fc(hidden[-1])


def collate_fn(batch):
    texts, labels = zip(*batch)
    texts_padded = pad_sequence(texts, batch_first=True, padding_value=0)  # 填充文本序列
    labels = torch.stack(labels)  # 将标签堆叠成一个张量
    return texts_padded, labels


def train(model, dataloader, optimizer, criterion, device):
    model.train()
    for texts, labels in dataloader:
        texts, labels = texts.to(device), labels.to(device)
        optimizer.zero_grad()
        predictions = model(texts)
        loss = criterion(predictions, labels)
        loss.backward()
        optimizer.step()


def classify_emotion(model, tokenizer, text, device, dataset):
    model.eval()
    with torch.no_grad():
        tokens = tokenizer.texts_to_sequences([text])
        tokens = pad_sequence([torch.tensor(seq, dtype=torch.long) for seq in tokens], batch_first=True)
        tokens = tokens.to(device)
        predictions = model(tokens)
        emotion_index = torch.argmax(predictions, dim=1).item()
    return dataset.label_encoder.inverse_transform([emotion_index])[0]


def generate_response(emotion):
    return random.choice(responses[emotion])


def main():

    VOCAB_SIZE = 1000
    EMBEDDING_DIM = 64
    HIDDEN_DIM = 128
    OUTPUT_DIM = len(responses)
    N_LAYERS = 2
    DROPOUT = 0.5
    BATCH_SIZE = 2
    EPOCHS = 10
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    tokenizer = SimpleTokenizer()
    tokenizer.fit_on_texts(texts)
    dataset = TextDataset(texts, labels, tokenizer, max_len=20)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)  # 使用自定义 collate_fn


    model = EmotionClassifier(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, N_LAYERS, DROPOUT).to(DEVICE)
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()


    for epoch in range(EPOCHS):
        train(model, dataloader, optimizer, criterion, DEVICE)
        print(f"Epoch {epoch + 1}/{EPOCHS} completed.")


    print("Hello! I'm your empathetic AI assistant. How are you feeling today?")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "bye", "goodbye"]:
            print("AI: Goodbye! I hope you have a wonderful day!")
            break
        emotion = classify_emotion(model, tokenizer, user_input, DEVICE, dataset)  # 传递 dataset 参数
        response = generate_response(emotion)
        print(f"AI: {response}")

if __name__ == "__main__":
    main()
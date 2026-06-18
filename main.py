import os
import re
import sys
import json
import time
import pickle
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report

from utils import preprocess_review

# --- 1. Load va split du lieu ---
def load_data():
    print("Dang doc file du lieu Shopee...")
    df_raw = pd.read_json("shopee_reviews_dataset.jsonl", lines=True)
    df_raw['sentiment'] = df_raw['label'].map({'positive': 1, 'negative': 0})
    df = df_raw[['review', 'sentiment']].rename(columns={'review': 'comment'}).copy()
    
    # Chia tap du lieu: 80% train, 10% val, 10% test
    df_train_full, df_test = train_test_split(df, test_size=0.1, random_state=42, stratify=df['sentiment'])
    df_train, df_val = train_test_split(df_train_full, test_size=0.1111, random_state=42, stratify=df_train_full['sentiment'])
    
    print(f"So luong tap train: {len(df_train)}, val: {len(df_val)}, test: {len(df_test)}")
    return df_train, df_val, df_test

# --- 2. Tien xu ly ---
def preprocess_dataset(df_train, df_val, df_test):
    print("Dang tien xu ly van ban (word segmenting, teencode)...")
    df_train['clean_comment'] = df_train['comment'].apply(preprocess_review)
    df_val['clean_comment'] = df_val['comment'].apply(preprocess_review)
    df_test['clean_comment'] = df_test['comment'].apply(preprocess_review)
    
    # Loc dong trong
    df_train = df_train[df_train['clean_comment'] != ""].dropna()
    df_val = df_val[df_val['clean_comment'] != ""].dropna()
    df_test = df_test[df_test['clean_comment'] != ""].dropna()
    
    return df_train, df_val, df_test

# --- 3. Build Vocab cho LSTM ---
def build_vocab(df_train):
    from collections import Counter
    all_words = []
    for text in df_train['clean_comment']:
        all_words.extend(text.split())
    
    vocab_counter = Counter(all_words)
    # Loc tu tan suat >= 2
    filtered_words = [word for word, count in vocab_counter.items() if count >= 2]
    vocab = {word: idx + 2 for idx, word in enumerate(filtered_words)}
    vocab['<PAD>'] = 0
    vocab['<UNK>'] = 1
    return vocab

# --- 4. Dataset cho PyTorch ---
class ReviewDataset(Dataset):
    def __init__(self, df, vocab, max_len=80):
        self.labels = df['sentiment'].values
        self.sequences = []
        for text in df['clean_comment']:
            seq = [vocab.get(w, vocab['<UNK>']) for w in text.split()]
            # Padding hoac cat
            if len(seq) < max_len:
                seq = seq + [0] * (max_len - len(seq))
            else:
                seq = seq[:max_len]
            self.sequences.append(seq)
            
    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, idx):
        return torch.tensor(self.sequences[idx], dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.float)

# --- 5. Mo hinh Bi-LSTM ---
class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim=100, hidden_dim=128, output_dim=1, n_layers=2, dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, 
                           hidden_dim, 
                           num_layers=n_layers, 
                           bidirectional=True, 
                           batch_first=True,
                           dropout=dropout if n_layers > 1 else 0)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text):
        embedded = self.dropout(self.embedding(text))
        output, (hidden, cell) = self.lstm(embedded)
        pooled = torch.mean(output, dim=1) # Mean pooling
        return self.fc(self.dropout(pooled))

# --- 6. Huan luyen model ---
def train_baselines(df_train, df_test):
    print("\n--- HUAN LUYEN BASELINES (TF-IDF) ---")
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=10000)
    X_train_tfidf = vectorizer.fit_transform(df_train['clean_comment'])
    X_test_tfidf = vectorizer.transform(df_test['clean_comment'])
    y_train = df_train['sentiment'].values
    y_test = df_test['sentiment'].values
    
    # Naive Bayes
    t0 = time.time()
    nb_model = MultinomialNB()
    nb_model.fit(X_train_tfidf, y_train)
    nb_time = time.time() - t0
    y_pred_nb = nb_model.predict(X_test_tfidf)
    
    # Logistic Regression
    t0 = time.time()
    lr_model = LogisticRegression(max_iter=1000)
    lr_model.fit(X_train_tfidf, y_train)
    lr_time = time.time() - t0
    y_pred_lr = lr_model.predict(X_test_tfidf)
    
    # Danh gia
    nb_acc = accuracy_score(y_test, y_pred_nb)
    nb_f1 = f1_score(y_test, y_pred_nb, average='macro')
    lr_acc = accuracy_score(y_test, y_pred_lr)
    lr_f1 = f1_score(y_test, y_pred_lr, average='macro')
    
    # Save files
    with open('nb_model.pkl', 'wb') as f:
        pickle.dump(nb_model, f)
    with open('lr_model.pkl', 'wb') as f:
        pickle.dump(lr_model, f)
    with open('tfidf_vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
        
    return nb_acc, nb_f1, nb_time, lr_acc, lr_f1, lr_time, nb_model, lr_model, vectorizer

def train_lstm(df_train, df_val, df_test, vocab):
    print("\n--- HUAN LUYEN PYTORCH BI-LSTM (Lau hon) ---")
    max_len = 80
    train_dataset = ReviewDataset(df_train, vocab, max_len)
    val_dataset = ReviewDataset(df_val, vocab, max_len)
    test_dataset = ReviewDataset(df_test, vocab, max_len)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device huan luyen:", device)
    
    vocab_size = len(vocab)
    model = BiLSTMClassifier(vocab_size=vocab_size).to(device)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    epochs = 5 # Dat 5 de chay nhanh hon
    best_val_loss = float('inf')
    
    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        correct = 0
        total = 0
        for seqs, labels in train_loader:
            seqs, labels = seqs.to(device), labels.to(device)
            optimizer.zero_grad()
            predictions = model(seqs).squeeze(1)
            loss = criterion(predictions, labels)
            
            preds = torch.round(torch.sigmoid(predictions))
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        train_loss = epoch_loss / len(train_loader)
        train_acc = correct / total
        
        # Val
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for seqs, labels in val_loader:
                seqs, labels = seqs.to(device), labels.to(device)
                predictions = model(seqs).squeeze(1)
                loss = criterion(predictions, labels)
                
                preds = torch.round(torch.sigmoid(predictions))
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
                val_loss += loss.item()
                
        valid_loss = val_loss / len(val_loader)
        valid_acc = val_correct / val_total
        print(f"Epoch {epoch+1:02} | Train Loss: {train_loss:.3f} Acc: {train_acc*100:.2f}% | Val Loss: {valid_loss:.3f} Acc: {valid_acc*100:.2f}%")
        
        if valid_loss < best_val_loss:
            best_val_loss = valid_loss
            torch.save(model.state_dict(), 'best_lstm.pt')
            
    lstm_time = time.time() - t0
    
    # Test model
    model.load_state_dict(torch.load('best_lstm.pt'))
    model.eval()
    test_preds = []
    y_test = df_test['sentiment'].values
    with torch.no_grad():
        for seqs, _ in test_loader:
            seqs = seqs.to(device)
            predictions = model(seqs).squeeze(1)
            preds = torch.round(torch.sigmoid(predictions))
            test_preds.extend(preds.cpu().numpy())
            
    test_preds = np.array(test_preds)
    lstm_acc = accuracy_score(y_test, test_preds)
    lstm_f1 = f1_score(y_test, test_preds, average='macro')
    
    return lstm_acc, lstm_f1, lstm_time, model

# --- 7. Interactive Tester ---
def interactive_loop(lr_model, vectorizer, lstm_model, vocab, device):
    max_len = 80
    print("\n" + "="*50)
    print("CHUONG TRINH TUONG TAC REVIEW (REVIEW TESTER)")
    print("Ghi chu: Nen nhap review tieng Viet co dau vi mo hinh train tren data co dau.")
    print("Nhap 'exit' hoac 'thoat' de dung lai.")
    print("="*50)
    
    while True:
        try:
            review_text = input("\nNhap review cua ban: ").strip()
            if not review_text or review_text.lower() in ['exit', 'thoat']:
                break
                
            cleaned = preprocess_review(review_text)
            print(f"Sau tien xu ly: {cleaned}")
            
            # 1. Logistic Regression (TF-IDF)
            tfidf_vec = vectorizer.transform([cleaned])
            lr_pred = lr_model.predict(tfidf_vec)[0]
            lr_prob = lr_model.predict_proba(tfidf_vec)[0]
            lr_label = "Tich cuc" if lr_pred == 1 else "Tieu cuc"
            lr_confidence = lr_prob[1] if lr_pred == 1 else lr_prob[0]
            
            # 2. PyTorch Bi-LSTM
            seq = [vocab.get(w, vocab['<UNK>']) for w in cleaned.split()]
            if len(seq) < max_len:
                seq = seq + [0] * (max_len - len(seq))
            else:
                seq = seq[:max_len]
                
            lstm_model.eval()
            with torch.no_grad():
                tensor_seq = torch.tensor([seq], dtype=torch.long).to(device)
                logit = lstm_model(tensor_seq).squeeze(1).item()
                prob = torch.sigmoid(torch.tensor(logit)).item()
                lstm_label = "Tich cuc" if prob >= 0.5 else "Tieu cuc"
                lstm_confidence = prob if prob >= 0.5 else (1.0 - prob)
                
            print("-" * 50)
            print(f"Logistic Regression: {lr_label:<10} (Do tin cay: {lr_confidence*100:.2f}%)")
            print(f"PyTorch Bi-LSTM:     {lstm_label:<10} (Do tin cay: {lstm_confidence*100:.2f}%)")
            print("-" * 50)
        except KeyboardInterrupt:
            break
    print("\nDa thoat khoi chuong trinh tester!")

# --- Main Flow ---
if __name__ == '__main__':
    df_train, df_val, df_test = load_data()
    df_train, df_val, df_test = preprocess_dataset(df_train, df_val, df_test)
    
    vocab = build_vocab(df_train)
    vocab_size = len(vocab)
    print("Kich thuoc vocab:", vocab_size)
    with open('vocab.json', 'w', encoding='utf-8') as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
        
    nb_acc, nb_f1, nb_time, lr_acc, lr_f1, lr_time, nb_model, lr_model, vectorizer = train_baselines(df_train, df_test)
    lstm_acc, lstm_f1, lstm_time, lstm_model = train_lstm(df_train, df_val, df_test, vocab)
    
    # Hien thi bang so sanh
    print("\n" + "="*60)
    print("KET QUA SO SANH CAC MO HINH")
    print("="*60)
    print(f"{'Mo hinh':<30} | {'Accuracy (%)':<12} | {'F1-Score (%)':<12} | {'Time (s)':<8}")
    print("-" * 70)
    print(f"{'Naive Bayes (Baseline)':<30} | {nb_acc*100:<12.2f} | {nb_f1*100:<12.2f} | {nb_time:<8.4f}")
    print(f"{'Logistic Regression (Baseline)':<30} | {lr_acc*100:<12.2f} | {lr_f1*100:<12.2f} | {lr_time:<8.4f}")
    print(f"{'PyTorch Bi-LSTM (Advanced)':<30} | {lstm_acc*100:<12.2f} | {lstm_f1*100:<12.2f} | {lstm_time:<8.2f}")
    print("="*60)
    
    # Chay thu nghiem review
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    interactive_loop(lr_model, vectorizer, lstm_model, vocab, device)

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
from transformers import AutoModelForSequenceClassification, AutoTokenizer

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
    
    # Tinh toan do dai cau (so tu) de phan tich
    df_train['len_raw'] = df_train['comment'].apply(lambda x: len(str(x).split()))
    df_train['len_clean'] = df_train['clean_comment'].apply(lambda x: len(str(x).split()))
    
    print("\n--- PHAN TICH DO DAI CAU (TREN TAP TRAIN) ---")
    print(f"Do dai trung binh (truoc tien xu ly): {df_train['len_raw'].mean():.2f} tu")
    print(f"Do dai trung binh (sau tien xu ly):  {df_train['len_clean'].mean():.2f} tu")
    print(f"Do dai cau lon nhat (sau tien xu ly): {df_train['len_clean'].max()} tu")
    print(f"Do dai cau nho nhat (sau tien xu ly): {df_train['len_clean'].min()} tu")
    
    # Loc dong trong va loc cau qua ngan (duoi 2 tu) de giam nhieu (noise)
    df_train = df_train[df_train['len_clean'] >= 2].dropna()
    df_val = df_val[df_val['clean_comment'].apply(lambda x: len(str(x).split())) >= 2].dropna()
    df_test = df_test[df_test['clean_comment'].apply(lambda x: len(str(x).split())) >= 2].dropna()
    
    print(f"So luong du lieu sau khi loc cau ngan (< 2 tu): {len(df_train)}")
    
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

# --- 4.b Dataset cho Transformers ---
class TransformerDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.texts)
        
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'targets': torch.tensor(label, dtype=torch.long)
        }

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

# --- 6.b Huan luyen PhoBERT ---
def train_phobert(df_train, df_val, df_test):
    print("\n--- HUAN LUYEN PHO-BERT (HuggingFace) ---")
    print("LUU Y: Huan luyen PhoBERT tren CPU rat cham. Giam kich thuoc tap du lieu xuong 10% de demo.")
    
    df_train_sub = df_train.sample(frac=0.1, random_state=42)
    df_val_sub = df_val.sample(frac=0.1, random_state=42)
    df_test_sub = df_test.sample(frac=0.1, random_state=42)
    
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base", use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained("vinai/phobert-base", num_labels=2)
    
    max_len = 128
    batch_size = 16
    
    train_dataset = TransformerDataset(df_train_sub['clean_comment'].values, df_train_sub['sentiment'].values, tokenizer, max_len)
    val_dataset = TransformerDataset(df_val_sub['clean_comment'].values, df_val_sub['sentiment'].values, tokenizer, max_len)
    test_dataset = TransformerDataset(df_test_sub['clean_comment'].values, df_test_sub['sentiment'].values, tokenizer, max_len)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device huan luyen PhoBERT:", device)
    model = model.to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=2e-5)
    epochs = 1 # Demo 1 epoch
    
    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            targets = batch['targets'].to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=targets)
            loss = outputs.loss
            
            preds = torch.argmax(outputs.logits, dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)
            
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            
        print(f"Epoch {epoch+1} | Train Loss: {total_loss / len(train_loader):.3f} Acc: {correct/total*100:.2f}%")
        
    phobert_time = time.time() - t0
    
    # Test model
    model.eval()
    test_preds = []
    y_test_sub = df_test_sub['sentiment'].values
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1)
            test_preds.extend(preds.cpu().numpy())
            
    phobert_acc = accuracy_score(y_test_sub, test_preds)
    phobert_f1 = f1_score(y_test_sub, test_preds, average='macro')
    
    torch.save(model.state_dict(), 'best_phobert.pt')
    
    return phobert_acc, phobert_f1, phobert_time, model, tokenizer

# --- 6.c Huan luyen DistilBERT (Nhanh hon vi nhe hon) ---
def train_distilbert(df_train, df_val, df_test):
    print("\n--- HUAN LUYEN DISTILBERT (HuggingFace) ---")
    print("LUU Y: Chay tren 10% du lieu de demo nhanh.")
    
    df_train_sub = df_train.sample(frac=0.1, random_state=42)
    df_val_sub = df_val.sample(frac=0.1, random_state=42)
    df_test_sub = df_test.sample(frac=0.1, random_state=42)
    
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-multilingual-cased")
    model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-multilingual-cased", num_labels=2)
    
    max_len = 128
    batch_size = 16
    
    train_dataset = TransformerDataset(df_train_sub['clean_comment'].values, df_train_sub['sentiment'].values, tokenizer, max_len)
    val_dataset = TransformerDataset(df_val_sub['clean_comment'].values, df_val_sub['sentiment'].values, tokenizer, max_len)
    test_dataset = TransformerDataset(df_test_sub['clean_comment'].values, df_test_sub['sentiment'].values, tokenizer, max_len)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device huan luyen DistilBERT:", device)
    model = model.to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=3e-5)
    epochs = 1
    
    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            targets = batch['targets'].to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=targets)
            loss = outputs.loss
            
            preds = torch.argmax(outputs.logits, dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)
            
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            
        print(f"Epoch {epoch+1} | Train Loss: {total_loss / len(train_loader):.3f} Acc: {correct/total*100:.2f}%")
        
    distilbert_time = time.time() - t0
    
    # Test model
    model.eval()
    test_preds = []
    y_test_sub = df_test_sub['sentiment'].values
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1)
            test_preds.extend(preds.cpu().numpy())
            
    distilbert_acc = accuracy_score(y_test_sub, test_preds)
    distilbert_f1 = f1_score(y_test_sub, test_preds, average='macro')
    
    torch.save(model.state_dict(), 'best_distilbert.pt')
    
    return distilbert_acc, distilbert_f1, distilbert_time, model, tokenizer

# --- 6.d Danh gia Pre-trained Sentiment PhoBERT (0s training!) ---
def eval_pretrained_sentiment(df_test):
    print("\n--- DANH GIA PRE-TRAINED SENTIMENT MODEL (Khong can train) ---")
    print("Su dung model: wonrax/phobert-base-vietnamese-sentiment")
    
    df_test_sub = df_test.sample(frac=0.1, random_state=42)
    
    tokenizer = AutoTokenizer.from_pretrained("wonrax/phobert-base-vietnamese-sentiment", use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained("wonrax/phobert-base-vietnamese-sentiment")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    test_preds = []
    y_test_sub = df_test_sub['sentiment'].values
    
    t0 = time.time()
    with torch.no_grad():
        for text in df_test_sub['clean_comment'].values:
            inputs = tokenizer(text, return_tensors="pt", max_length=128, padding='max_length', truncation=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)
            # 0: NEG, 1: NEU, 2: POS. Chuyen ve Binary: POS > NEG -> 1, nguoc lai -> 0
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]
            neg_prob = probs[0].item()
            pos_prob = probs[2].item()
            pred = 1 if pos_prob > neg_prob else 0
            test_preds.append(pred)
            
    eval_time = time.time() - t0
    
    acc = accuracy_score(y_test_sub, test_preds)
    f1 = f1_score(y_test_sub, test_preds, average='macro')
    
    return acc, f1, eval_time, model, tokenizer

# --- 7. Interactive Tester ---
def interactive_loop(lr_model, vectorizer, lstm_model, vocab, phobert_model, phobert_tokenizer, wonrax_model, wonrax_tokenizer, device):
    max_len_lstm = 80
    max_len_phobert = 128
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
            if len(seq) < max_len_lstm:
                seq = seq + [0] * (max_len_lstm - len(seq))
            else:
                seq = seq[:max_len_lstm]
                
            lstm_model.eval()
            with torch.no_grad():
                tensor_seq = torch.tensor([seq], dtype=torch.long).to(device)
                logit = lstm_model(tensor_seq).squeeze(1).item()
                prob = torch.sigmoid(torch.tensor(logit)).item()
                lstm_label = "Tich cuc" if prob >= 0.5 else "Tieu cuc"
                lstm_confidence = prob if prob >= 0.5 else (1.0 - prob)
                
            # 3. PhoBERT
            phobert_model.eval()
            encoding = phobert_tokenizer(
                cleaned,
                add_special_tokens=True,
                max_length=max_len_phobert,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt',
            )
            with torch.no_grad():
                input_ids = encoding['input_ids'].to(device)
                attention_mask = encoding['attention_mask'].to(device)
                outputs = phobert_model(input_ids=input_ids, attention_mask=attention_mask)
                probs = torch.nn.functional.softmax(outputs.logits, dim=1)
                phobert_confidence, phobert_pred = torch.max(probs, dim=1)
                phobert_label = "Tich cuc" if phobert_pred.item() == 1 else "Tieu cuc"
                phobert_confidence = phobert_confidence.item()
                
            # 4. Pre-trained Sentiment PhoBERT (wonrax)
            wonrax_model.eval()
            inputs_wonrax = wonrax_tokenizer(cleaned, return_tensors="pt", max_length=max_len_phobert, padding='max_length', truncation=True)
            inputs_wonrax = {k: v.to(device) for k, v in inputs_wonrax.items()}
            with torch.no_grad():
                outputs_wonrax = wonrax_model(**inputs_wonrax)
                probs_wonrax = torch.nn.functional.softmax(outputs_wonrax.logits, dim=-1)[0]
                neg_p = probs_wonrax[0].item()
                pos_p = probs_wonrax[2].item()
                wonrax_label = "Tich cuc" if pos_p > neg_p else "Tieu cuc"
                wonrax_confidence = max(pos_p, neg_p) / (pos_p + neg_p + 1e-9)

            print("-" * 75)
            print(f"Logistic Regression:         {lr_label:<10} (Do tin cay: {lr_confidence*100:.2f}%)")
            print(f"PyTorch Bi-LSTM:             {lstm_label:<10} (Do tin cay: {lstm_confidence*100:.2f}%)")
            print(f"PhoBERT (10% Fine-tuned):    {phobert_label:<10} (Do tin cay: {phobert_confidence*100:.2f}%)")
            print(f"PhoBERT (Pre-trained Full):  {wonrax_label:<10} (Do tin cay: {wonrax_confidence*100:.2f}%)")
            print("-" * 75)
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
    phobert_acc, phobert_f1, phobert_time, phobert_model, phobert_tokenizer = train_phobert(df_train, df_val, df_test)
    distil_acc, distil_f1, distil_time, distil_model, distil_tokenizer = train_distilbert(df_train, df_val, df_test)
    pre_acc, pre_f1, pre_time, pre_model, pre_tokenizer = eval_pretrained_sentiment(df_test)
    
    # Hien thi bang so sanh
    print("\n" + "="*80)
    print("KET QUA SO SANH CAC MO HINH")
    print("="*80)
    print(f"{'Mo hinh':<35} | {'Accuracy (%)':<12} | {'F1-Score (%)':<12} | {'Time (s)':<8}")
    print("-" * 90)
    print(f"{'Naive Bayes (Baseline)':<35} | {nb_acc*100:<12.2f} | {nb_f1*100:<12.2f} | {nb_time:<8.4f}")
    print(f"{'Logistic Regression (Baseline)':<35} | {lr_acc*100:<12.2f} | {lr_f1*100:<12.2f} | {lr_time:<8.4f}")
    print(f"{'PyTorch Bi-LSTM (Advanced)':<35} | {lstm_acc*100:<12.2f} | {lstm_f1*100:<12.2f} | {lstm_time:<8.2f}")
    print(f"{'PhoBERT (10% Fine-tuned)':<35} | {phobert_acc*100:<12.2f} | {phobert_f1*100:<12.2f} | {phobert_time:<8.2f}")
    print(f"{'DistilBERT (10% Fine-tuned)':<35} | {distil_acc*100:<12.2f} | {distil_f1*100:<12.2f} | {distil_time:<8.2f}")
    print(f"{'PhoBERT (Pre-trained Full - 0s train)':<35} | {pre_acc*100:<12.2f} | {pre_f1*100:<12.2f} | {pre_time:<8.2f}")
    print("="*80)
    
    # Chay thu nghiem review
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    interactive_loop(lr_model, vectorizer, lstm_model, vocab, phobert_model, phobert_tokenizer, pre_model, pre_tokenizer, device)

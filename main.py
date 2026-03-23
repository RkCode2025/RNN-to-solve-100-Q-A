import torch
import torch.nn as nn
import pandas as pd
import os
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

# --- 1. Load and Clean Data ---
def load_data():
    # Define the relative path to the data
    data_path = os.path.join("..", "data", "100_Unique_QA_Dataset.csv")

    # Check if file exists, fallback to local directory if not
    if not os.path.exists(data_path):
        data_path = "100_Unique_QA_Dataset.csv"

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Could not find {data_path}. Please ensure the CSV is in the correct folder.")

    print(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path)
    df.columns = ['question', 'answer']
    return df

# --- 2. Tokenization and Vocabulary ---
def tokenizer(text):
    text = str(text).lower()
    text = text.replace("?", "").replace("'", "")
    return text.split()

def build_vocab(df):
    vocab = {'<UNK>': 0}
    for _, row in df.iterrows():
        tokens = tokenizer(row["question"]) + tokenizer(row["answer"])
        for token in tokens:
            if token not in vocab:
                vocab[token] = len(vocab)
    return vocab

def text_to_indices(text, vocab):
    return [vocab.get(token, vocab["<UNK>"]) for token in tokenizer(text)]

# --- 3. Dataset and DataLoader ---
class QADataloader(Dataset):
    def __init__(self, df, vocab):
        self.df = df
        self.vocab = vocab

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        num_ques = text_to_indices(self.df.iloc[index]['question'], self.vocab)
        num_ans = text_to_indices(self.df.iloc[index]['answer'], self.vocab)
        return torch.tensor(num_ques), torch.tensor(num_ans)

def collate_fn(batch):
    questions, answers = zip(*batch)
    ques_padded = pad_sequence(questions, batch_first=True, padding_value=0)
    ans_padded = pad_sequence(answers, batch_first=True, padding_value=0)
    return ques_padded, ans_padded

# --- 4. Model Definition ---
class SimpleRNN(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embedding_dim=50)
        self.rnn = nn.RNN(50, 64, batch_first=True)
        self.fc = nn.Linear(64, vocab_size)

    def forward(self, question):
        embed_ques = self.embed(question)
        # hn: [num_layers, batch, hidden_size]
        output, hn = self.rnn(embed_ques)
        # Using the last hidden state: hn.squeeze(0) works for 1-layer RNN
        return self.fc(hn.squeeze(0))

# --- 5. Prediction Function ---
def predict(model, question, vocab, inv_vocab, threshold=0.5):
    model.eval()
    num_ques = text_to_indices(question, vocab)
    torch_ques = torch.tensor(num_ques).unsqueeze(0)
    
    with torch.no_grad():
        output = model(torch_ques)
        probability = torch.nn.functional.softmax(output, dim=1)
        max_prob, index = torch.max(probability, dim=1)
        
    if max_prob.item() < threshold:
        return "I don't know."
    
    return inv_vocab.get(index.item(), "<UNK>")

# --- 6. Main Execution ---
if __name__ == "__main__":
    # Load Data
    df = load_data()
    
    # Setup Vocab
    vocab = build_vocab(df)
    inv_vocab = {v: k for k, v in vocab.items()}
    print(f"Vocab size: {len(vocab)}")

    # Setup DataLoader
    dataset = QADataloader(df, vocab)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)

    # Initialize Model
    model = SimpleRNN(len(vocab))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Training
    epochs = 20
    print("Starting training...")
    for epoch in range(epochs):
        total_loss = 0
        model.train()
        for question, answer in dataloader:
            optimizer.zero_grad()
            
            pred = model(question)
            target = answer[:, 0].long()
            
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch: {epoch+1}/{epochs} | Loss: {total_loss/len(dataloader):.4f}")

    # Test Prediction
    test_q = "WHAT IS 5+5"
    ans = predict(model, test_q, vocab, inv_vocab)
    print(f"\nTest Query: {test_q}")
    print(f"Model Prediction: {ans}")

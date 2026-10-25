"""Stage 4: BiLSTM + Attention + IBPNN (improved back-propagation network head)."""
import numpy as np
import torch
import torch.nn as nn

class AuditNet(nn.Module):
    def __init__(self, hidden=16):
        super().__init__()
        self.lstm = nn.LSTM(1, hidden, batch_first=True, bidirectional=True)  # BiLSTM over the 9 indicators
        self.att = nn.Linear(2 * hidden, 1)                                   # attention over indicators
        self.head = nn.Sequential(nn.Linear(2 * hidden, 16), nn.Tanh(), nn.Linear(16, 1), nn.Sigmoid())

    def forward(self, x):
        out, _ = self.lstm(x.unsqueeze(-1))
        w = torch.softmax(self.att(out).squeeze(-1), dim=1)
        return self.head((out * w.unsqueeze(-1)).sum(1)).squeeze(-1), w

def train(X, target, epochs=30, seed=7, max_rows=20000):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))[:max_rows]
    Xt, Tt = torch.tensor(X[idx], dtype=torch.float32), torch.tensor(target[idx], dtype=torch.float32)
    model = AuditNet()
    # IBPNN: back-propagation with momentum-style adaptive steps (Adam) and plateau learning-rate decay
    opt = torch.optim.Adam(model.parameters(), lr=5e-3)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=.5, patience=2)
    for _ in range(epochs):
        total = 0.0
        for b in torch.randperm(len(Xt)).split(256):
            opt.zero_grad()
            loss = nn.functional.mse_loss(model(Xt[b])[0], Tt[b])
            loss.backward()
            opt.step()
            total += loss.item() * len(b)
        sched.step(total / len(Xt))
    return model.eval()

def predict(model, X):
    with torch.no_grad():
        p, w = model(torch.tensor(np.asarray(X), dtype=torch.float32))
    return p.numpy().astype(float), w.numpy()

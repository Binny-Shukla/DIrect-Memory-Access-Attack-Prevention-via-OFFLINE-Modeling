import torch


class offline_learning_pipeline:
    """Offline-only supervised update step from reviewed, labelled event features."""

    def train_epoch(self, model, loader, optimizer):
        model.train()
        loss_fn = torch.nn.CrossEntropyLoss()
        total_loss = 0.0
        for features, labels in loader:
            logits = model(features)
            loss = loss_fn(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        return total_loss / max(len(loader), 1)

import torch


def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, model_path, best_model_path):
    """Train and validate the model with best model saving and detailed logging."""
    best_val_acc = 0.0
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    print(f"Starting training on device: {device}")
    print(f"Number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    for epoch in range(num_epochs):
        print(f"\n--- Epoch {epoch + 1}/{num_epochs} ---")

        # Training loop
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        num_batches = 0

        for batch_idx, batch in enumerate(train_loader):
            frames = batch['frames'].to(device)
            group_labels = batch['group_label'].to(device)

            optimizer.zero_grad()
            logits = model(frames)
            loss = criterion(logits, group_labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = logits.max(1)
            total += group_labels.size(0)
            correct += predicted.eq(group_labels).sum().item()
            num_batches += 1

            if batch_idx % 10 == 0:  # Debug print every 10 batches
                print(f"  Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")

        avg_train_loss = train_loss / num_batches
        train_acc = 100. * correct / total
        train_losses.append(avg_train_loss)
        train_accs.append(train_acc)
        print(f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%")

        # Validation loop
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        num_batches = 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                frames = batch['frames'].to(device)
                group_labels = batch['group_label'].to(device)
                logits = model(frames)
                loss = criterion(logits, group_labels)
                val_loss += loss.item()
                _, predicted = logits.max(1)
                total += group_labels.size(0)
                correct += predicted.eq(group_labels).sum().item()
                num_batches += 1

                if batch_idx % 5 == 0:  # Debug print every 5 batches
                    print(f"  Val Batch {batch_idx}/{len(val_loader)}, Loss: {loss.item():.4f}")

        avg_val_loss = val_loss / num_batches
        val_acc = 100. * correct / total
        val_losses.append(avg_val_loss)
        val_accs.append(val_acc)
        print(f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"  New best model saved with Val Acc: {best_val_acc:.2f}%")

    # Save final model
    torch.save(model.state_dict(), model_path)
    print(f"\nTraining completed. Final model saved to {model_path}")
    print(f"Best model saved to {best_model_path} with Val Acc: {best_val_acc:.2f}%")

    return model, train_losses, val_losses, train_accs, val_accs
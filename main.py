# main.py
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

from data_indexing import merge_datasets
from dataset import VolleyballDataset
from model import VolleyballBaseline1
from train import train_model
from visualization import create_visualization_report, plot_training_history
from config import (
    VID_DIR, FRAMES_PATH, LABELS_PATH, MERGED_PATH, MODEL_PATH,
    TRAIN_IDX, VAL_IDX, TEST_IDX, TRAIN_TRANSFORM, VAL_TEST_TRANSFORM, BATCH_SIZE, NUM_WORKERS,
    NUM_EPOCHS, LEARNING_RATE, WEIGHT_DECAY, BACKBONE
)


def main():
    # Step 1: Merge datasets
    merged_dataset = merge_datasets(
        LABELS_PATH, FRAMES_PATH, MERGED_PATH, TRAIN_IDX, VAL_IDX, TEST_IDX
    )

    # Step 2: Create datasets
    train_dataset = VolleyballDataset(
        merged_dataset['train'], transform=TRAIN_TRANSFORM,
        person_level=False, single_frame=True
    )
    val_dataset = VolleyballDataset(
        merged_dataset['val'], transform=VAL_TEST_TRANSFORM,
        person_level=False, single_frame=True
    )
    test_dataset = VolleyballDataset(
        merged_dataset['test'], transform=VAL_TEST_TRANSFORM,
        person_level=False, single_frame=True
    )

    # Step 3: DataLoaders
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True
    )

    # Step 4: Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_group_classes = len(train_dataset.group_to_idx)
    model = VolleyballBaseline1(num_group_classes=num_group_classes, backbone_name=BACKBONE).to(device)
    print(f"✅ Model created with backbone={BACKBONE}, trainable params={sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    # Step 5: Loss & Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    # Step 6: Train model
    best_model_path = MODEL_PATH.replace(".pth", "_best.pth")
    model, train_losses, val_losses, train_accs, val_accs = train_model(
        model, train_loader, val_loader, criterion, optimizer,
        num_epochs=NUM_EPOCHS, device=device,
        model_path=MODEL_PATH, best_model_path=best_model_path
    )

    # Step 7: Visualization report

    class_names = list(train_dataset.group_to_idx.keys())
    plot_training_history(train_losses, val_losses, train_accs, val_accs, save_path="training_history.png")
    print("\n📊 Generating visualization report...")
    create_visualization_report(merged_dataset, model, test_loader, class_names)


if __name__ == "__main__":
    main()

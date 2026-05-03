# =============================================================
# step2_train_model.py  —  ResNet18 training + Grad-CAM + SHAP
# Run AFTER step1_prepare_data.py
# =============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, ConfusionMatrixDisplay)
import shap
import warnings
warnings.filterwarnings('ignore')

from config import *


# ── 1. Dataset class ─────────────────────────────────────────
class OASISDataset(Dataset):
    def __init__(self, records, transform=None):
        self.records   = records.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        row   = self.records.iloc[idx]
        img   = Image.open(row['png_path']).convert('RGB')
        label = int(row['Label'])
        if self.transform:
            img = self.transform(img)
        return img, label


# ── 2. Transforms ─────────────────────────────────────────────
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])


# ── 3. Build model ────────────────────────────────────────────
def build_model():
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    # freeze all layers except last block + FC
    for name, param in model.named_parameters():
        if 'layer4' not in name and 'fc' not in name:
            param.requires_grad = False

    # replace final FC layer for 3 classes
    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(model.fc.in_features, NUM_CLASSES)
    )
    return model


# ── 4. Training loop ──────────────────────────────────────────
def train_model(model, train_loader, val_loader, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE
    )
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    history = {'train_loss': [], 'val_loss': [],
               'train_acc':  [], 'val_acc':  []}

    best_val_acc = 0.0
    best_model_path = os.path.join(MODEL_DIR, "best_resnet18.pth")

    print(f"\nTraining on: {device}")
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")
    print("-" * 50)

    for epoch in range(NUM_EPOCHS):
        # ── train ──
        model.train()
        t_loss, t_correct, t_total = 0.0, 0, 0

        for imgs, labels in tqdm(train_loader,
                                 desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]",
                                 leave=False):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss    = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            t_loss    += loss.item() * imgs.size(0)
            preds      = outputs.argmax(dim=1)
            t_correct += (preds == labels).sum().item()
            t_total   += imgs.size(0)

        # ── validate ──
        model.eval()
        v_loss, v_correct, v_total = 0.0, 0, 0

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                loss    = criterion(outputs, labels)
                v_loss    += loss.item() * imgs.size(0)
                preds      = outputs.argmax(dim=1)
                v_correct += (preds == labels).sum().item()
                v_total   += imgs.size(0)

        train_loss = t_loss / t_total
        val_loss   = v_loss / v_total
        train_acc  = t_correct / t_total
        val_acc    = v_correct / v_total

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        print(f"Epoch {epoch+1:02d}/{NUM_EPOCHS} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"  ✓ Best model saved (val_acc={val_acc:.4f})")

        scheduler.step()

    print(f"\nBest validation accuracy: {best_val_acc:.4f}")
    return history, best_model_path


# ── 5. Evaluate on test set ───────────────────────────────────
def evaluate(model, test_loader, device):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            probs   = torch.softmax(outputs, dim=1).cpu().numpy()
            preds   = outputs.argmax(dim=1).cpu().numpy()
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    all_probs  = np.array(all_probs)
    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    print("\n" + "="*60)
    print("TEST SET EVALUATION")
    print("="*60)
    print(classification_report(all_labels, all_preds,
                                target_names=LABEL_NAMES))

    # AUC (one-vs-rest)
    try:
        auc = roc_auc_score(all_labels, all_probs, multi_class='ovr')
        print(f"AUC-ROC (macro OvR): {auc:.4f}")
    except Exception as e:
        print(f"AUC could not be computed: {e}")

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(cm, display_labels=LABEL_NAMES)
    fig, ax = plt.subplots(figsize=(7, 6))
    disp.plot(ax=ax, colorbar=False, cmap='Blues')
    ax.set_title("Confusion Matrix — Test Set")
    plt.tight_layout()
    save_path = os.path.join(RESULTS_DIR, "confusion_matrix.png")
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"Confusion matrix saved: {save_path}")

    return all_preds, all_labels, all_probs


# ── 6. Plot training history ──────────────────────────────────
def plot_history(history):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history['train_loss'], label='Train Loss')
    axes[0].plot(history['val_loss'],   label='Val Loss')
    axes[0].set_title('Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].legend()

    axes[1].plot(history['train_acc'], label='Train Acc')
    axes[1].plot(history['val_acc'],   label='Val Acc')
    axes[1].set_title('Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].legend()

    plt.tight_layout()
    save_path = os.path.join(RESULTS_DIR, "training_history.png")
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"Training history saved: {save_path}")


# ── 7. Grad-CAM ───────────────────────────────────────────────
class GradCAM:
    def __init__(self, model, target_layer):
        self.model        = model
        self.target_layer = target_layer
        self.gradients    = None
        self.activations  = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor, class_idx=None):
        self.model.eval()
        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        self.model.zero_grad()
        output[0, class_idx].backward()

        weights = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam     = (weights * self.activations).sum(dim=1, keepdim=True)
        cam     = torch.relu(cam)
        cam     = cam.squeeze().numpy()

        # normalise
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())

        return cam, class_idx


def apply_gradcam(model, test_records, device, n_samples=6):
    print("\nGenerating Grad-CAM visualisations...")
    gradcam = GradCAM(model, model.layer4[-1].conv2)

    samples = test_records.sample(min(n_samples, len(test_records)),
                                  random_state=42)
    fig, axes = plt.subplots(2, n_samples, figsize=(n_samples * 3, 6))
    fig.suptitle("Grad-CAM: Original vs Heatmap", fontsize=13)

    for i, (_, row) in enumerate(samples.iterrows()):
        img_pil = Image.open(row['png_path']).convert('RGB')
        img_t   = val_transform(img_pil).unsqueeze(0)

        cam, pred_class = gradcam.generate(img_t)

        # resize cam to IMG_SIZE
        cam_img = Image.fromarray((cam * 255).astype(np.uint8))
        cam_img = cam_img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
        cam_np  = np.array(cam_img) / 255.0

        # overlay
        heatmap   = cm.jet(cam_np)[:, :, :3]
        orig_np   = np.array(img_pil.resize((IMG_SIZE, IMG_SIZE))) / 255.0
        overlay   = 0.5 * orig_np + 0.5 * heatmap

        axes[0][i].imshow(orig_np, cmap='gray')
        axes[0][i].set_title(f"True: {row['Group']}\nCDR={row['CDR']}", fontsize=8)
        axes[0][i].axis('off')

        axes[1][i].imshow(overlay)
        axes[1][i].set_title(f"Pred: {LABEL_NAMES[pred_class]}", fontsize=8)
        axes[1][i].axis('off')

    plt.tight_layout()
    save_path = os.path.join(RESULTS_DIR, "gradcam_results.png")
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"Grad-CAM saved: {save_path}")


# ── 8. SHAP ───────────────────────────────────────────────────
def apply_shap(model, train_records, test_records, device):
    print("\nGenerating SHAP explanations...")
    model.eval()
    model.to('cpu')

    def load_batch(records, n=30):
        imgs = []
        for _, row in records.sample(min(n, len(records)),
                                     random_state=42).iterrows():
            img = val_transform(
                Image.open(row['png_path']).convert('RGB')
            )
            imgs.append(img)
        return torch.stack(imgs)

    # ── Method 1: SHAP GradientExplainer ─────────────────────
    try:
        background  = load_batch(train_records, n=30)
        test_imgs   = load_batch(test_records,  n=6)

        explainer   = shap.GradientExplainer(model, background)
        shap_values = explainer.shap_values(test_imgs)

        fig, axes = plt.subplots(1, NUM_CLASSES, figsize=(14, 4))
        fig.suptitle("SHAP Feature Importance by Class", fontsize=13)

        for c in range(NUM_CLASSES):
            sv = np.array(shap_values[c])  # shape: (n, C, H, W) or (n, H, W)

            # force to (n, C, H, W)
            if sv.ndim == 3:
                sv = sv[:, np.newaxis, :, :]

            # average over samples and channels → (H, W)
            mean = np.abs(sv).mean(axis=0)  # → (C, H, W)
            mean = mean.mean(axis=0)  # → (H, W)

            # ensure it is square — reshape if needed
            if mean.shape[0] != mean.shape[1]:
                side = int(np.sqrt(mean.size))
                mean = mean.flatten()[:side * side].reshape(side, side)

            axes[c].imshow(mean, cmap='hot', aspect='auto')
            axes[c].set_title(f"Class: {LABEL_NAMES[c]}", fontsize=10)
            axes[c].axis('off')



            axes[c].set_title(f"Class: {LABEL_NAMES[c]}", fontsize=10)
            axes[c].axis('off')

        plt.tight_layout()
        save_path = os.path.join(RESULTS_DIR, "shap_results.png")
        plt.savefig(save_path, dpi=150)
        plt.show()
        print(f"SHAP saved: {save_path}")

    except Exception as e:
        print(f"  SHAP GradientExplainer failed: {e}")
        print("  Falling back to gradient saliency maps...")

        # ── Method 2: Gradient saliency fallback ─────────────
        try:
            test_imgs = load_batch(test_records, n=6)
            test_imgs.requires_grad_(True)

            model.zero_grad()
            outputs = model(test_imgs)

            # compute saliency for predicted class per image
            saliency_maps = []
            for i in range(test_imgs.shape[0]):
                model.zero_grad()
                if test_imgs.grad is not None:
                    test_imgs.grad.zero_()
                score = outputs[i].max()
                score.backward(retain_graph=True)
                sal = test_imgs.grad[i].abs().mean(dim=0).detach().numpy()
                if sal.max() > sal.min():
                    sal = (sal - sal.min()) / (sal.max() - sal.min())
                saliency_maps.append(sal)

            n_show = len(saliency_maps)
            cols   = min(3, n_show)
            rows   = (n_show + cols - 1) // cols
            fig, axes = plt.subplots(rows, cols,
                                     figsize=(cols * 4, rows * 4))
            if rows == 1 and cols == 1:
                axes = [[axes]]
            elif rows == 1:
                axes = [axes]
            fig.suptitle("Gradient Saliency Maps (XAI Fallback)", fontsize=13)

            sample_rows = test_records.head(n_show).reset_index(drop=True)
            for i, sal in enumerate(saliency_maps):
                ax  = axes[i // cols][i % cols]
                ax.imshow(sal, cmap='hot')
                row = sample_rows.iloc[i]
                ax.set_title(f"True: {row['Group']}\nCDR={row['CDR']}",
                             fontsize=9)
                ax.axis('off')

            # hide unused subplots
            for j in range(n_show, rows * cols):
                axes[j // cols][j % cols].axis('off')

            plt.tight_layout()
            save_path = os.path.join(RESULTS_DIR, "shap_results.png")
            plt.savefig(save_path, dpi=150)
            plt.show()
            print(f"Saliency map saved: {save_path}")

        except Exception as e2:
            print(f"  Saliency fallback also failed: {e2}")
            print("  Skipping XAI — all other results are saved.")


# ── MAIN ─────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*60)
    print("STEP 2: ResNet18 Training + XAI")
    print("="*60)

    # load master index from step 1
    index_path = os.path.join(OUTPUT_DIR, "master_index.csv")
    assert os.path.exists(index_path), \
        "master_index.excel not found! Run step1_prepare_data.py first."

    df = pd.read_csv(index_path)
    print(f"\nLoaded {len(df)} records from master index")

    # train / val / test split (subject-level to avoid leakage)
    subjects   = df['Subject ID'].unique()
    train_subs, temp_subs = train_test_split(
        subjects, test_size=(1 - TRAIN_SPLIT), random_state=RANDOM_SEED)
    val_subs, test_subs   = train_test_split(
        temp_subs, test_size=0.5, random_state=RANDOM_SEED)

    train_df = df[df['Subject ID'].isin(train_subs)]
    val_df   = df[df['Subject ID'].isin(val_subs)]
    test_df  = df[df['Subject ID'].isin(test_subs)]

    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    # data loaders
    train_ds = OASISDataset(train_df, train_transform)
    val_ds   = OASISDataset(val_df,   val_transform)
    test_ds  = OASISDataset(test_df,  val_transform)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=0)

    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # build and train — skip if best model already exists
    best_model_path = os.path.join(MODEL_DIR, "best_resnet18.pth")
    model = build_model().to(device)

    if os.path.exists(best_model_path):
        print(f"\nFound existing model at {best_model_path}")
        print("Skipping training — loading saved model directly.")
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        history = None
    else:
        history, best_model_path = train_model(model, train_loader,
                                               val_loader, device)
        plot_history(history)

    # load best model and evaluate
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    preds, labels, probs = evaluate(model, test_loader, device)

    # save predictions alongside subject info for ODE pipeline
    test_df = test_df.copy()
    test_df['Predicted_Label'] = preds
    test_df['Pred_Nondemented'] = probs[:, 0]
    test_df['Pred_Converted']   = probs[:, 1]
    test_df['Pred_Demented']    = probs[:, 2]
    pred_path = os.path.join(RESULTS_DIR, "test_predictions.csv")
    test_df.to_csv(pred_path, index=False)
    print(f"\nTest predictions saved: {pred_path}")

    # XAI
    apply_gradcam(model, test_df, device)
    apply_shap(model, train_df, test_df, device)

    print("\n✓ Step 2 complete. Run step3_ode_model.py next.")

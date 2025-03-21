# Import library yang diperlukan
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Path dataset
train_dataset_path = "./DATASET/TRAINING"
test_dataset_path = "./DATASET/TESTING"

# Definisi kelas dataset
class ImageTensorDataset(Dataset):
    def __init__(self, dataset_path, size=(224, 224)):
        self.image_tensors = []
        self.labels = []
        self.class_to_idx = {}  # Mapping class ke indeks
        current_label = 0
        transform = transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor(),  # Konversi ke tensor [0, 1]
        ])
        for person in os.listdir(dataset_path):
            person_path = os.path.join(dataset_path, person)
            if not os.path.isdir(person_path):
                continue
            if person not in self.class_to_idx:
                self.class_to_idx[person] = current_label
                current_label += 1
            for img_file in os.listdir(person_path):
                if img_file.lower().endswith((".png", ".jpg", ".jpeg")):
                    img_path = os.path.join(person_path, img_file)
                    img = Image.open(img_path).convert("RGB")  # Baca gambar
                    img_tensor = transform(img)  # Resize & convert ke tensor
                    self.image_tensors.append(img_tensor)
                    self.labels.append(self.class_to_idx[person])
        self.image_tensors = torch.stack(self.image_tensors)
        self.labels = torch.tensor(self.labels)

    def __len__(self):
        return len(self.image_tensors)

    def __getitem__(self, idx):
        return self.image_tensors[idx], self.labels[idx]

# Load dataset
train_dataset = ImageTensorDataset(train_dataset_path)
test_dataset = ImageTensorDataset(test_dataset_path)

print(f"Jumlah data pelatihan: {len(train_dataset)}")
print(f"Jumlah data pengujian: {len(test_dataset)}")

# Buat DataLoader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print(f"Jumlah batch pelatihan: {len(train_loader)}")
print(f"Jumlah batch pengujian: {len(test_loader)}")

# Inisialisasi parameter model
input_size = 224 * 224 * 3  # Flatten dari citra RGB ukuran 224x224
num_classes = len(train_dataset.class_to_idx)  # Jumlah kelas
idx_to_class = {v: k for k, v in train_dataset.class_to_idx.items()}  # Mapping indeks ke nama folder

print(f"Input size: {input_size}")
print(f"Number of classes: {num_classes}")
print(f"Index to class mapping: {idx_to_class}")

# Definisi model ANN
class ANNModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(ANNModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 512)  # Fully connected layer 1
        self.fc2 = nn.Linear(512, 256)  # Fully connected hidden layer 2
        self.fc3 = nn.Linear(256, num_classes)  # Output layer

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten input
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Trainer untuk model
class ANNTrainer:
    def __init__(self, model, train_loader, test_loader, criterion=nn.CrossEntropyLoss(), lr=0.01,
                 optimizer_type="SGD", device=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.criterion = criterion.to(self.device)
        if optimizer_type == "SGD":
            self.optimizer = optim.SGD(self.model.parameters(), lr=lr)
        elif optimizer_type == "MiniBatchGD":
            self.optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)
        elif optimizer_type == "GD":
            self.optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=0)
        else:
            raise ValueError("Optimizer type not recognized")

    def train(self, epochs=10):
        self.model.train()
        training_errors = []
        for epoch in range(epochs):
            total_loss = 0
            for images, labels in self.train_loader:
                images, labels = images.to(self.device), labels.to(self.device)  # Pindahkan ke GPU/CPU
                images = images.view(images.size(0), -1)  # Flatten gambar sebelum masuk model
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(self.train_loader)
            training_errors.append(avg_loss)
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}")
        return training_errors

    def test(self):
        self.model.eval()
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for images, labels in self.test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                images = images.view(images.size(0), -1)
                outputs = self.model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        accuracy = 100 * correct / total
        print(f"Testing Accuracy: {accuracy:.2f}%")
        return accuracy, all_preds, all_labels

# Set up parameter dan inisialisasi model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
optimizer_type = "SGD"
model = ANNModel(input_size, num_classes).to(device)
trainer = ANNTrainer(model, train_loader, test_loader, optimizer_type=optimizer_type)
epochs = 1000

print(f"Using device: {device}")
print(f"Using optimizer: {optimizer_type}")

# Training model
model.train()
training_errors = trainer.train(epochs=epochs)

# Save the trained model
torch.save(model.state_dict(), "ann_model.pth")  # Save the model's state dict

print("Model has been saved to ann_model.pth")

# Evaluasi dengan data training
model.eval()
train_results = []
correct = 0
total = 0
all_train_preds = []
all_train_labels = []

with torch.no_grad():
    for i in range(len(train_dataset)):
        img_tensor, label = train_dataset[i]
        img_tensor = img_tensor.to(device)
        img_tensor = img_tensor.reshape(-1, input_size)
        output = model(img_tensor)
        _, predicted = torch.max(output, 1)
        pred_label = idx_to_class[predicted.item()]
        true_label = idx_to_class[label.item()]
        file_name = f"train_image_{i}.png"
        train_results.append([file_name, pred_label, true_label])
        total += 1
        correct += (predicted.item() == label.item())
        all_train_preds.append(predicted.item())
        all_train_labels.append(label.item())

train_accuracy = correct / total
train_error_rate = 1 - train_accuracy
print(f"Train Accuracy: {train_accuracy:.4f}, Train Error Rate: {train_error_rate:.4f}")

# Simpan hasil training ke CSV
pd.DataFrame(train_results, columns=["File Name", "Predicted Label", "Actual Label"]).to_csv("train_results.csv", index=False)

# Testing model
accuracy, all_preds, all_labels = trainer.test()

test_results = []
with torch.no_grad():
    for i in range(len(test_dataset)):
        img_tensor, label = test_dataset[i]
        img_tensor = img_tensor.to(device)
        img_tensor = img_tensor.reshape(-1, input_size)
        output = model(img_tensor)
        _, predicted = torch.max(output, 1)
        pred_label = idx_to_class[predicted.item()]
        true_label = idx_to_class[label.item()]
        file_name = f"test_image_{i}.png"
        test_results.append([file_name, pred_label, true_label])

# Simpan hasil testing ke CSV
pd.DataFrame(test_results, columns=["File Name", "Predicted Label", "Actual Label"]).to_csv("test_results.csv", index=False)

# Evaluasi model dengan data testing
precision = precision_score(all_labels, all_preds, average='macro')
recall = recall_score(all_labels, all_preds, average='macro')
f1 = f1_score(all_labels, all_preds, average='macro')
print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")

# Plot error training
plt.figure(figsize=(10, 5))
plt.plot(range(1, epochs + 1), training_errors, marker='o', linestyle='-')
plt.xlabel("Epoch")
plt.ylabel("Training Loss")
plt.title("Training Error over Epochs")
plt.show()

print("Hasil prediksi data training disimpan di train_results.csv")
print("Hasil prediksi data testing disimpan di test_results.csv")

# MENAMPILKAN GAMBAR HASIL TESTING
num_images = 23  # Jumlah gambar yang ingin ditampilkan (5x5 grid)
rows, cols = 5, 5  # Grid untuk menampilkan gambar

# Buat subplot dengan grid 5x5
fig, axes = plt.subplots(rows, cols, figsize=(15, 15))

# Iterasi untuk menampilkan gambar
for i in range(num_images):
    img_tensor, label = test_dataset[i]
    img = img_tensor.permute(1, 2, 0).cpu().numpy()

    # Prediksi label
    img_tensor = img_tensor.to(device)
    img_tensor = img_tensor.reshape(-1, input_size)
    output = model(img_tensor)
    _, predicted = torch.max(output, 1)
    pred_label = idx_to_class[predicted.item()]
    true_label = idx_to_class[label.item()]

    # Hitung posisi row dan col berdasarkan indeks i
    row = i // cols
    col = i % cols

    # Plot gambar pada grid
    axes[row, col].imshow(img)
    axes[row, col].set_title(f"True: {true_label}\nPredicted: {pred_label}", fontsize=8)
    axes[row, col].axis('off')

# Jika jumlah gambar kurang dari grid, sembunyikan subplot yang tidak digunakan
if num_images < rows * cols:
    for i in range(num_images, rows * cols):
        row = i // cols
        col = i % cols
        axes[row, col].axis('off')

plt.tight_layout()
plt.show()
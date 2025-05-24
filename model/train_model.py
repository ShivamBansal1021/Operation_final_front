import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T

class GraphDataset(Dataset):

    def __init__(self, csv_file, images_dir, transform=None):

        import csv
        self.graph_data = []
        with open(csv_file, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)  
            for row in reader:
                graph_id = row[0]
                num_nodes = int(row[1])

                adjacency_values = list(map(int, row[2:2+36]))
                self.graph_data.append((graph_id, num_nodes, adjacency_values))
        self.images_dir = images_dir
        self.transform = transform

    def __len__(self):
        return len(self.graph_data)

    def __getitem__(self, idx):

        graph_id, num_nodes, adjacency_values = self.graph_data[idx]

        img_path = os.path.join(self.images_dir, f"{graph_id}.png")
        img = Image.open(img_path).convert('RGB')

        if self.transform:
            img = self.transform(img)
        else:
            img = img.resize((128, 128))
            img = T.ToTensor()(img)

        node_count_label = num_nodes - 1

        adjacency_label = torch.tensor(adjacency_values, dtype=torch.float32)
        return img, node_count_label, adjacency_label

class GraphNet(nn.Module):

    def __init__(self):
        super(GraphNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(64 * 8 * 8, 128)

        self.fc_count = nn.Linear(128, 6)
        self.fc_adj = nn.Linear(128, 36)
        self.relu = nn.ReLU()

    def forward(self, x):

        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.relu(self.conv3(x))
        x = self.pool(x)
        x = self.relu(self.conv4(x))
        x = self.pool(x)

        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
 
        count_out = self.fc_count(x)
        adj_out = self.fc_adj(x)
        return count_out, adj_out

if __name__ == "__main__":

    this_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(this_dir, os.pardir))
    csv_path = os.path.join(project_root, "data", "adjacency_matrices.csv")
    images_dir = os.path.join(project_root, "data", "graphs_images")

    dataset = GraphDataset(csv_path, images_dir)
    train_loader = DataLoader(dataset, batch_size=16, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GraphNet().to(device)
    criterion_count = nn.CrossEntropyLoss()
    criterion_adj = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # Training loop
    num_epochs = 100
    model.train()
    for epoch in range(1, num_epochs + 1):
        running_loss = 0.0
        for images, count_labels, adj_labels in train_loader:
            images = images.to(device)
            count_labels = count_labels.to(device)
            adj_labels = adj_labels.to(device)
            optimizer.zero_grad()

            count_logits, adj_logits = model(images)

            loss_count = criterion_count(count_logits, count_labels)
            loss_adj = criterion_adj(adj_logits, adj_labels)
            loss = loss_count + loss_adj

            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch}/{num_epochs}, Loss: {epoch_loss:.4f}")

    model_path = os.path.join(this_dir, "graph_model.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Training complete. Model saved to {model_path}")

import os
import torch
from PIL import Image
import torchvision.transforms as T
from model.train_model import GraphNet

def predict_graph(image_path):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = GraphNet().to(device)
    model_dir = os.path.dirname(os.path.abspath(__file__))
    weights_path = os.path.join(model_dir, "graph_model.pth")
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    img = Image.open(image_path).convert('RGB')
    img = img.resize((128, 128))
    img_tensor = T.ToTensor()(img).unsqueeze(0).to(device)

    with torch.no_grad():
        count_logits, adj_logits = model(img_tensor)

        pred_class = torch.argmax(count_logits, dim=1)
        predicted_num_nodes = int(pred_class.item() + 1) 

        adj_probs = torch.sigmoid(adj_logits)
        adj_pred_flat = (adj_probs >= 0.5).int()
        adj_pred_matrix = adj_pred_flat.view(6, 6).tolist()
    return predicted_num_nodes, adj_pred_matrix

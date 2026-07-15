import torch
import torch.nn as nn
from torchvision import models
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
from torchvision import transforms
from PIL import Image
import cv2

def clasificacion_birads_micro(img):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # -----------------------------
    # 1. Cargar modelo
    # -----------------------------
    new_model = models.resnet50(weights="IMAGENET1K_V2")
    new_model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    new_model.fc = nn.Linear(new_model.fc.in_features, 2)
    state_dict = torch.load('final_model_weights_2.pth', map_location=torch.device('cpu'))
    new_model.load_state_dict(state_dict)
    new_model = new_model.to(device)
    new_model.eval() 

    # -----------------------------
    # 2. Transformación
    # -----------------------------

    IMG_SIZE = 512
    
    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.25])
    ])

    # -----------------------------
    # 3. Hooks para Grad-CAM
    # -----------------------------
    gradients = None
    activations = None

    def save_gradients(module, grad_in, grad_out):
        nonlocal gradients
        gradients = grad_out[0]

    def save_activations(module, input, output):
        nonlocal activations
        activations = output

    # Última capa convolucional de ResNet50
    target_layer = new_model.layer4[-1].conv3
    target_layer.register_forward_hook(save_activations)
    target_layer.register_backward_hook(save_gradients)

    # -----------------------------
    # 4. Predicción normal
    # -----------------------------
    
    def predict_image(model, img):
        model.eval()
    
        # 1. Cargar imagen en escala de grises (como en SimpleDataset)
        img = img.convert("L")
    
        # 2. Aplicar EXACTAMENTE el mismo preprocesado que val_transform
        img_tensor = val_transform(img).unsqueeze(0).to(device)  # [1, 1, 512, 512]
    
        # 3. Forward
        with torch.no_grad():
            outputs = model(img_tensor)
            probs = torch.softmax(outputs, dim=1)
            _, pred = torch.max(outputs, 1)
    
        return pred.item(), probs.cpu().numpy(), img_tensor

    #Y ahora la predicción propiamente dicha:
    pred, probs, img_tensor = predict_image(new_model, img)

    # -----------------------------
    # 5. Grad-CAM
    # -----------------------------
    # Hacer backward sobre la clase predicha
    new_model.zero_grad()
    score = new_model(img_tensor)[0, pred]
    score.backward()

    # Pesos: media espacial de gradientes
    weights = gradients.mean(dim=(2, 3), keepdim=True)

    # Mapa CAM
    cam = (weights * activations).sum(dim=1)
    cam = torch.relu(cam)
    cam = cam.squeeze().detach().cpu().numpy()

    # Normalizar
    cam = cam - cam.min()
    cam = cam / cam.max()

    # -----------------------------
    # 6. Superponer Grad-CAM sobre la imagen original
    # -----------------------------
    img_np = np.array(img.convert("L").resize((IMG_SIZE, IMG_SIZE)))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    heatmap = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))

    overlay = (0.4 * heatmap + 0.6 * np.stack([img_np]*3, axis=-1)).astype(np.uint8)
    

    return pred, probs, overlay
    
    
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms
import gdown
import os
import streamlit as st
import cv2

# ============================================================
# 1. ARQUITECTURA U-NET v3 (exacta del Colab)
# ============================================================

class BloqueDobleConv(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),

            nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )
        for m in self.net:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        return self.net(x)


class BloqueAtencion(nn.Module):
    def __init__(self, canales):
        super().__init__()
        r = max(canales // 8, 1)

        self.avg = nn.AdaptiveAvgPool2d(1)
        self.mx = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(canales, r, bias=False),
            nn.ReLU(),
            nn.Linear(r, canales, bias=False)
        )

        self.sp = nn.Conv2d(2, 1, 7, padding=3, bias=False)

    def forward(self, x):
        w = torch.sigmoid(self.fc(self.avg(x)) + self.fc(self.mx(x)))
        x = x * w.unsqueeze(-1).unsqueeze(-1)

        sp = torch.sigmoid(self.sp(torch.cat([x.mean(1, True),
                                              x.max(1).values.unsqueeze(1)], 1)))
        return x * sp


class UNet(nn.Module):
    def __init__(self, cin=3, cout=1, features=[64,128,256,512,1024], dropout_p=0.3):
        super().__init__()

        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.atenciones = nn.ModuleList()
        self.pool = nn.MaxPool2d(2)

        # Down
        for f in features:
            self.downs.append(BloqueDobleConv(cin, f))
            cin = f

        # Bottleneck
        self.bottleneck = nn.Sequential(
            BloqueDobleConv(features[-1], features[-1]*2),
            nn.Dropout2d(p=dropout_p)
        )

        # Up
        for f in reversed(features):
            self.ups.append(nn.ConvTranspose2d(f*2, f, 2, stride=2))
            self.atenciones.append(BloqueAtencion(f))
            self.ups.append(BloqueDobleConv(f*2, f))

        self.final_conv = nn.Conv2d(features[0], cout, 1)
        nn.init.constant_(self.final_conv.bias, -2.0)

    def forward(self, x):
        skips = []

        for down in self.downs:
            x = down(x)
            skips.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skips = skips[::-1]

        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x)
            skip = self.atenciones[i//2](skips[i//2])

            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)

            x = self.ups[i+1](torch.cat([skip, x], dim=1))

        return self.final_conv(x)


# ============================================================
# 2. PREPROCESADO EXACTO DEL COLAB
# ============================================================

IMG_SIZE = 256

infer_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],
                         std=[0.229,0.224,0.225])
])



# ============================================================
# 3. FUNCIÓN DE INFERENCIA
# ============================================================

def segmentacion_unet(img):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- Cargar pesos ----
    nombre_pesos = "pesos_unet_v3.pth"
    drive_id = "1LrhPHpyMfNQAk6YLv6xlxILyRtXN01lD"
    url = f"https://drive.google.com/uc?export=download&id={drive_id}"

    if not os.path.exists(nombre_pesos):
        st.warning('Descargando el modelo U-Net de internet (~400 MB). Puede tardar un rato, no cierres la aplicación.')
        with st.spinner("Descargando modelo..."):
            gdown.download(url, nombre_pesos, quiet=False)
        st.success('Modelo descargado con éxito')

    # ---- Crear modelo ----
    model = UNet(cin=3, cout=1).to(device)

    state_dict = torch.load(nombre_pesos, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # ---- Preprocesado ----
    img_np = np.array(img)
    h_original, w_original = img_np.shape[:2]
    
    img = img.convert("RGB")
    tensor = infer_transform(img).unsqueeze(0).to(device)

    # ---- Forward ----
    with torch.no_grad():
        pred = torch.sigmoid(model(tensor))
        pred_np = pred.squeeze().cpu().numpy()

    # ---- Binarización ----
    mask_bin = (pred_np > 0.5).astype(np.uint8)
    mask_img = Image.fromarray(mask_bin * 255)
    mask_img = np.array(mask_img).astype(np.uint8)

    #La revertimos a su resolución original
    mascara_revertida = cv2.resize(
        mask_img,
        (w_original, h_original),  # (width, height)
        interpolation=cv2.INTER_NEAREST
    )

    return mascara_revertida, pred_np

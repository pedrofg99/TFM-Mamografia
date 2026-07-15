def segmentacion_ML(img):
    from sklearn.mixture import GaussianMixture
    import numpy as np
    import os
    from PIL import Image
    import matplotlib.pyplot as plt
    import cv2

    #Definimos el preprocesado
    def preprocess_ML(img):
        # 1) Percentile clipping
        p1, p99 = np.percentile(img, (1, 99))
        img = np.clip(img, p1, p99)
    
        # 2) Normalización 0–1
        img = (img - p1) / (p99 - p1)
    
        # 3) Convertir a 8 bits
        img = (img * 255).astype(np.uint8)
    
        # 4) CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        img = clahe.apply(img)
    
        # 5) Padding a cuadrado
        h, w = img.shape
        size = max(h, w)
        padded = np.zeros((size, size), dtype=np.uint8)
        padded[:h, :w] = img

        # 6) Resize final
        img = cv2.resize(padded, (512, 512), interpolation=cv2.INTER_AREA)
    
        return img

    #Definimos función para revertir la máscara a la resolución original
    def revert_mask(mask_512, original_h, original_w):
        # 1) Revertir el resize: volver al tamaño del padded cuadrado
        size = max(original_h, original_w)
        mask_padded = cv2.resize(mask_512, (size, size), interpolation=cv2.INTER_NEAREST)
    
        # 2) Quitar el padding: recortar a la forma original
        mask_original = mask_padded[:original_h, :original_w]
    
        return mask_original
    
        
    #Definimos la función que segmenta por k-means
    def segment_gmm_intensity(img, K=3):
        h, w = img.shape
    
        X = img.reshape(-1, 1)  # solo intensidad
    
        gmm = GaussianMixture(n_components=K, covariance_type='full', random_state=0)
        gmm.fit(X)
    
        labels = gmm.predict(X).reshape(h, w)
    
        # Elegir el clúster más brillante
        means = gmm.means_.flatten()
        cluster_tumor = np.argmax(means)
    
        mask = (labels == cluster_tumor).astype(np.uint8)
    
        return mask, labels, means

        #Definimos las funciones de post-filtrado de la máscara:
    def opening_by_reconstruction(mask, iterations=1):
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    
        # erosión
        eroded = cv2.erode(mask, ker, iterations=iterations)
    
        # reconstrucción morfológica
        reconstructed = cv2.dilate(eroded, ker, iterations=iterations)
        reconstructed = np.minimum(reconstructed, mask)
    
        return reconstructed



    def compute_roundness(area, perimeter):
        if perimeter == 0:
            return 0
        return (4 * np.pi * area) / (perimeter ** 2)


    def clean_mask(mask, k = 1, alpha = 0.5, area_percent = 0.01):
        # 1) Apertura morfológica para quitar puntos sueltos
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
        mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
        # 2) Cierre para rellenar huecos internos
        mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel)
    
        # ⭐ 2.5) Opening by reconstruction para romper cuellos finos
        mask_clean = opening_by_reconstruction(mask_clean, iterations=1)
    
        # 3) Componentes conectadas
        num_labels, labels = cv2.connectedComponents(mask_clean)
    
        if num_labels <= 1:
            return mask_clean  # no hay regiones
    
        areas = []
        roundnesses = []
    
        # Calcular métricas por componente
        for lab in range(1, num_labels):
            comp = (labels == lab).astype(np.uint8)
    
            # Área
            area = comp.sum()
    
            # Perímetro
            cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(cnts) == 0:
                perimeter = 0
            else:
                perimeter = cv2.arcLength(cnts[0], True)
    
            roundness = compute_roundness(area, perimeter)
    
            areas.append(area)
            roundnesses.append(roundness)
    
        areas = np.array(areas, dtype=float)
        roundnesses = np.array(roundnesses, dtype=float)
    
        # -----------------------------
        # 🔥 ELIMINAR pegotes pequeños
        # -----------------------------
        total_area = (mask > 0).sum()
        min_area = area_percent * total_area
        
        valid = areas >= min_area
        if not np.any(valid):
            return np.zeros_like(mask, dtype=np.uint8)
    
        # Filtrar arrays
        areas_f = areas[valid]
        round_f = roundnesses[valid]
    
        # Guardar qué etiquetas sobreviven
        surviving_labels = np.where(valid)[0] + 1
        # -----------------------------
    
        # Normalizar área entre 0 y 1 (solo válidos)
        norm_area = (areas_f - areas_f.min()) / (areas_f.max() - areas_f.min() + 1e-8)
    
        # Score combinado
        score = alpha * norm_area + (1 - alpha) * round_f
    
        # Elegir los k mejores entre los válidos
        idx_top = np.argsort(score)[::-1][:k]
        labels_top = surviving_labels[idx_top]
    
        # Construir máscara final
        final_mask = np.isin(labels, labels_top).astype(np.uint8)
    
        return final_mask

    #Ahora definimos ya lo que hace, propiamente, nuestra función:
    img = np.array(img)
    h_original, w_original = img.shape[:2]
    # Asegurar que la imagen es 2D (grayscale)
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    #Preprocesamos
    img_prepro = preprocess_ML(img)
    #Aplicamos la segmentación
    mascara, imagen_clusters, medias = segment_gmm_intensity(img_prepro, K = 4)

    #Filtramos la máscara
    mascara_limpia = clean_mask(mascara, k = 4, alpha = 0.2, area_percent = 0.01)

    #La revertimos a su resolución original
    mascara_revertida = revert_mask(mascara_limpia, h_original, w_original)

    return mascara_revertida
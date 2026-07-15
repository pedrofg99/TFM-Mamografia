def clasificacion_birads_mass(roi):
#Primero definimos las funciones necesarias
    import numpy as np
    import cv2
    from skimage.feature import graycomatrix, graycoprops
    from scipy.stats import skew, kurtosis
    from skimage.measure import regionprops
    from skimage.feature import local_binary_pattern
    import joblib
    import shap
    
    def extract_features(img, mask=None):
        """
        Extrae features globales de una mamografía:
        - Intensidad global
        - Histograma de intensidades (32 bins)
        - Textura GLCM
        - Gradientes globales
        - Morfología (si hay máscara)
        """
    
        # Asegurar escala de grises y tipo uint8
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = img.astype(np.uint8)
    
        feats = []
    
        # -----------------------------
        # 1. INTENSIDAD GLOBAL
        # -----------------------------
        vals = img[img > 0].reshape(-1) #Esto elimina los píxeles negros del fondo, para que no estropeen
        #las features
    
        img[img == 0] = vals.mean()
        #Aquí sustituimos los píxeles negros de img por píxeles que sean la media del resto
    
        feats.append(vals.mean())
        feats.append(vals.std())
        feats.append(skew(vals))
        feats.append(kurtosis(vals))
        p10, p50, p90 = np.percentile(vals, [10, 50, 90])
        feats.extend([p10, p50, p90])
    
        # -----------------------------
        # 2. HISTOGRAMA (32 bins)
        # -----------------------------
        hist, _ = np.histogram(vals, bins=32, range=(0, 255), density=True)
        feats.extend(hist.tolist())
    
        # -----------------------------
        # 3. TEXTURA GLCM GLOBAL
        # -----------------------------
        # Redimensionar para que GLCM no sea enorme
        img_small = cv2.resize(img, (256, 256))
    
        glcm = graycomatrix(img_small,
                            distances=[1],
                            angles=[0],
                            levels=256,
                            symmetric=True,
                            normed=True)
    
        feats.append(graycoprops(glcm, 'contrast')[0, 0])
        feats.append(graycoprops(glcm, 'homogeneity')[0, 0])
        feats.append(graycoprops(glcm, 'energy')[0, 0])
        feats.append(graycoprops(glcm, 'correlation')[0, 0])
    
        # -----------------------------
        # 4. GRADIENTES GLOBALES
        # -----------------------------
        Gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        Gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        Gmag = np.sqrt(Gx**2 + Gy**2)
        Lap = cv2.Laplacian(img, cv2.CV_64F)
    
        feats.append(Gmag.mean())
        feats.append(Gmag.std())
        feats.append(Lap.mean())
        feats.append(Lap.std())
    
        # -------------------------------
        # 5. --- LBP (Local Binary Patterns) ---
        #------------------------------------
    
        # ...
        lbp = local_binary_pattern(img, P=8, R=1, method="uniform")
        
        # P+2 = 10 patrones → 10 bins
        lbp_hist, _ = np.histogram(lbp.ravel(),
                                   bins=10,
                                   range=(0, 10),
                                   density=True)
        
        feats.extend(lbp_hist.tolist())
    
        # -----------------------------
        # 5. MORFOLOGÍA (si hay máscara)
        # -----------------------------
        if mask is not None:
            mask_bin = (mask > 0).astype(np.uint8)
    
            props = regionprops(mask_bin.astype(int))
            if len(props) > 0:
                r = props[0]
    
                area = r.area
                perimeter = r.perimeter if r.perimeter > 0 else 1
                circularity = 4 * np.pi * area / (perimeter**2)
                eccentricity = r.eccentricity
                solidity = r.solidity
    
                feats.extend([area, perimeter, circularity, eccentricity, solidity])
            else:
                feats.extend([0, 0, 0, 0, 0])
    
        return np.array(feats, dtype=float)

    #FUNCION PREPROCESADO ROIs
#Las ROIs no van a cambiarse de resolución, van a ir tal cual están. Solo aplicaremos algunas cosas como CLAHE
    def preprocess_ML_ROI(img):
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
    
        return img

    #Ahora ponemos la función propiamente dicha
    #Primero preprocesamos el roi:
    roi_pre = preprocess_ML_ROI(roi)
    #Ahora extraemos sus features:
    X = extract_features(roi_pre)
    X = X.reshape(1, -1)
    #Y ahora aplicamos el modelo, importándolo
    svm = joblib.load("modelo_birads_mass.pkl")
    probs = svm.predict_proba(X) #Te da un array con la prob de cada clase

    #Ahora mostramos el análisis SVM
        #Calculamos SHAP VALUES
    # Cargar background generado en el entrenamiento
    X_train_background = np.load("background_clasifmass.npy")
    background = shap.kmeans(X_train_background, 10)
    # Crear explainer (solo una vez)
    explainer = shap.KernelExplainer(svm.predict, background)
    # Calcular SHAP para esta ROI
    
    shap_values = explainer.shap_values(X)[0] 

    # base_values debe ser array de 1 elemento
    base_values = np.array([explainer.expected_value])             # (1,)
    
    # data debe ser 2D: (1, n_features)
    data = X.reshape(1, -1)                                    # (1, n_features)
    #Creamos los nombres de las features:
    #Creamos un array con los nombres de las features:
    feature_names = []
    
    # 1. Intensidad global
    feature_names += ["mean", "std", "skew", "kurtosis", "p10", "p50", "p90"]
    
    # 2. Histograma (32 bins)
    feature_names += [f"hist_bin_{i}" for i in range(32)]
    
    # 3. GLCM
    feature_names += ["glcm_contrast", "glcm_homogeneity", "glcm_energy", "glcm_correlation"]
    
    # 4. Gradientes
    feature_names += ["grad_mag_mean", "grad_mag_std", "lap_mean", "lap_std"]
    
    # 5. LBP
    feature_names += [f"lbp_uniform_{i}" for i in range(10)]

    exp = shap.Explanation(
    values = shap_values,
    base_values = base_values,
    feature_names = feature_names
)
    
    return probs, exp

    
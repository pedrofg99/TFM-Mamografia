import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
from segmentacion_ML import segmentacion_ML
from streamlit_image_zoom import image_zoom
from streamlit_cropper import st_cropper
from PIL import Image
from clasificacion_MASSvsCALC import clasificacion_MASSvsCALC
from clasificacion_birads_mass import clasificacion_birads_mass
from clasificacion_birads_micro import clasificacion_birads_micro
import shap
import matplotlib.pyplot as plt
import numpy as np
from streamlit_file_browser import st_file_browser

st.title("Clasificador híbrido de lesiones mamarias")

tab_clasificacion, tab_ayuda = st.tabs(["Clasificación ROI", "Ayuda y documentación"])

with tab_clasificacion:

    carpeta = r"C:\Users\pedro\Documents\MÁSTER UNIR 💻\TFM\extracción de imágenes del dataset\Imagenes png INbreast RENOMBRADAS"
    carpeta = st.text_input("Introduce la ruta de la carpeta de imágenes:")

    
    if not carpeta:
        st.stop()

    if not os.path.isdir(carpeta):
        st.error("❌ La ruta no existe. Revisa que esté bien escrita.")
        st.stop()
    
   
    st.success("Carpeta válida ✔")
            
    files = os.listdir(carpeta)
       
    fname = st.selectbox("Selecciona una imagen", files)
    file_path = os.path.join(carpeta, fname)
    img = Image.open(file_path).convert("RGB")
    img_np = np.array(img)
    h_img, w_img = img_np.shape[:2]
    
    # Convertir la imagen NumPy a PIL
    img_pil = Image.fromarray(img_np)
        
    st.subheader("Selecciona el ROI")
        
    cropped_img = st_cropper(
        img_pil,
        realtime_update=True,      # actualiza mientras mueves el cuadro
        box_color="#00FF00",       # color del cuadro
        aspect_ratio=None          # proporción libre
    )
        
    st.subheader("ROI recortado:")
    st.image(cropped_img)
    
    usar_ml = st.selectbox(
        "¿Quieres ayudarte de una segmentación?",
        ("No", "Usar segmentación mediante clústering", 'Usar segmentación mediante red neuronal U-net (no disponible)')
    )
    #Posible ayuda de segmentación ML (o no):
    if usar_ml == 'Usar segmentación mediante clústering':
        mask = segmentacion_ML(img_np)  # tu función ML
        mask = mask.astype(np.uint8)
    
        # Si viene en 0–1, escálala
        if np.max(mask) <= 1:
            mask = (mask * 255).astype(np.uint8)
        
        mask_color = np.zeros_like(img_np)
        mask_color[:,:,1] = mask # canal verde
        overlay = cv2.addWeighted(img_np, 0.9, mask_color, 0.1, 0)
        preview = overlay.copy()
    
        image_zoom(preview, mode="scroll", size = (700,800), keep_aspect_ratio=False, zoom_factor=4.0, increment=0.2)

        st.info('Nota: Esta segmentación puede equivocarse. La elección final de región de interés debe basarse exclusivamente en tu criterio')
    
    
    # ============================
    # BOTÓN: SOLO CALCULA MASSvsCALC UNA VEZ
    # ============================
    # Preprocesado ROI
    cropped_img_np = np.array(cropped_img)
    if len(cropped_img_np.shape) == 3:
        cropped_img_np = cv2.cvtColor(cropped_img_np, cv2.COLOR_RGB2GRAY)
    cropped_img_np = cropped_img_np.astype(np.uint8)
    
    if st.button("¿Este ROI contiene una masa?"):
    
        # MASS vs CALC SOLO UNA VEZ
        probs, exp = clasificacion_MASSvsCALC(cropped_img_np)
        probs_individual = probs[0]
        st.subheader(f'Probabilidad de ser masa: {(probs_individual[0]*100):.0f}%')

        st.subheader("Importancia de las características (SHAP)")
        
        fig, ax = plt.subplots()
        shap.plots.bar(exp, ax=ax)
        st.pyplot(fig)

        st.info('Las barras azules, de valores negativos, simbolizan que esas variables han fomentado que la decisión se incline hacia ser masa. Las barras rojas, de valores positivos, indican lo contrario.')
        st.info('Nota: haya o no haya una lesión en la región de interés proporcionada, el algoritmo de clasificación dará unos porcentajes. Eso no confirma que exista una lesión')
    
    
    tipo_clasificacion = st.selectbox(
        "¿Quieres clasificar esta imagen como masa o como microcalcificación?",
        ("Selecciona una opción", "Masa", "Microcalcificación"),
        index=0)
    
    if tipo_clasificacion == "Masa":
        probs_birads, exp2 = clasificacion_birads_mass(cropped_img_np)
        p0, p1 = probs_birads[0]
        st.subheader(f"Probabilidad benigno: {(p0*100):.0f}%")
        st.subheader(f"Probabilidad maligno: {(p1*100):.0f}%")

        st.subheader("Importancia de las características (SHAP)")
        
        fig, ax = plt.subplots()
        shap.plots.bar(exp2, ax=ax)
        st.pyplot(fig)

       
        st.info('Nota: haya o no haya una lesión en la región de interés proporcionada, el algoritmo de clasificación dará unos porcentajes. Eso no confirma que exista una lesión')
    
    elif tipo_clasificacion == "Microcalcificación":
        pred, probs2, overlay = clasificacion_birads_micro(img)
        p0, p1 = probs2[0]
        st.write(f"Probabilidad benigno: {(p0*100):.0f}%")
        st.write(f"Probabilidad maligno: {(p1*100):.0f}%")

        st.subheader('Mapa de activación (GRAD_CAM)')
        st.image(overlay, caption="Grad-CAM", use_column_width=True)

        
        st.info('Nota: Haya o no haya una lesión en la imagen proporcionada, el algoritmo de clasificación dará unos porcentajes. Eso no confirma que existan lesiones.')

with tab_ayuda:
    
    st.header("Ayuda y documentación")
    st.write('Este programa permite seleccionar una región de interés (ROI) dentro de una imagen mamográfica. Después, clasifica la imagen recortada según su grado de malignidad, como lesión benigna o lesión maligna. Además, contiene un algoritmo de segmentación que localiza posibles lesiones, con el fin de ayudar al usuario en la selección de la región de interés.')
    st.write('Este programa forma parte del Trabajo Fin de Estudios titulado "Clasificación Automatizada de Lesiones de Mama en Imágenes Médicas Empleando Técnicas Híbridas de Inteligencia Computacional y Aprendizaje Profundo", de autores Pedro Fernández González, Carmen Entremonzaga Melgosa y Ayman El Hichou Chahibou, dirigido por Ivón Oristela Benítez, que forma parte del máster en Ingeniería Matemática y Computación de la UNIR')

    st.write('Para acceder a la documentación detallada sobre este programa, así como conocer el código, por favor consulte el GITHUB, en el que se incluye el propio documento TFM')
             
    st.subheader('Importación de las imágenes y selección de ROI')
    st.write('En primer lugar, deberás indicar la carpeta en la que están tus imágenes mamográficas y seleccionar qué imagen quieres analizar en el menú desplegable. Una vez hecho, deberás seleccionar la región de interés utilizando un rectángulo interactivo')

    st.subheader('Segmentación de la imagen')
    st.write('Opcionalmente puedes ayudarte de una segmentación para elegir apropiadamente el ROI. Para ello, selecciona una opción en la selectbox titulada  "¿Quieres ayudarte de una segmentación?" Esto mostrará la imagen con la máscara de segmentación superpuesta, en color verde. Se puede hacer zoom libremente a esta imagen.')
    st.write('El algoritmo de segmentación con clústering primero preprocesa la imagen recortada o ROI para diferenciar mejor posibles lesiones, y luego utiliza un método de clústering sobre las intensidades de los píxeles, para separar los píxeles de la imagen en varios umbrales de intensidad. Se seleccionan los píxeles del umbral de mayor intensidad como candidatos a ser lesiones. Después, la máscara se post-procesa para limpiar ruido, separar regiones y quedarse solo con aquellas con un mejor equilibrio entre tamaño y redondez. Esas regiones se consideran posibles masas tumorales. Nótese que este algoritmo solo puede funcionar correctamente para masas, no para otro tipo de lesiones como podrían ser las microcalcificaciones. Además, no aseguramos que el algoritmo localice correctamente qué es una masa tumoral. Por tanto, esto debe tomarse exclusivamente como una posible ayuda, nunca como decisión final de lo que es tumor en la imagen mamográfica.')
    st.write('El algoritmo de segmentación con red neuronal U-net utiliza una red neuronal especializada para realizar la segmentación. Conviene ejecutar los dos y ver dónde coinciden. De todas formas, repetimos, ambos pueden fallar, por lo que la decisión última sobre qué es lesión en la imagen debe hacerla el usuario.')


    st.subheader('¿Este ROI contiene una masa?')
    st.write('En este programa, se hacen dos clasificaciones distintas: como masa o como microcalcificación. Esto es porque cada tipo de lesión necesita criterios distintos para ser clasificada como benigna o maligna. De esa manera, uno de los algoritmos es entrenado solo con imágenes que contienen lesiones tipo masa, y el otro clasificador se entrena solo con imágenes que contienen lesiones tipo microcalcificaciones. Por ello, antes de clasificar se permite, de forma opcional, hacer una clasificación extra que determine si el ROI contiene una masa o no. Eso puede ayudar a decidir, posteriormente, si hacer una clasificación como masa o como microcalcificación. El algoritmo, en primer lugar, preprocesa la imagen recortada o ROI para hacerla apta para los siguientes pasos. Después, transforma la imagen en un conjunto de datos, o features, relacionados con la distribución de intensidades, los gradientes (cambios de intensidad), y las texturas, entre otros. Ese conjunto de datos se pasa por un modelo de Machine Learning SVM (Support Vector Machine), que predice si la imagen es masa o microcalcificación. El modelo fue entrenado con ROIs de masas extraídos del dataset INbreast. Esos ROI se obtuvieron a partir de las máscaras de segmentación manuales hechas por los radiólogos del Dataset.')
    st.subheader('Clasificaciones para masas y para microcalcificaciones')
    st.write('El núcleo del programa son las dos clasificaciones finales: en ambas se clasifican las lesiones según su grado de malignidad, como BENIGNAS o MALIGNAS. BENIGNAS corresponden a aquellas con un bi-rads de INbreast menor que 4, y las MALIGNAS corresponden a un bi-rads mayor o igual a 4. En la selectbox "¿Quieres clasificar esta imagen como masa o como microcalcificación?" se puede seleccionar una de las dos opciones. La clasificación para masas usa el mismo algoritmo que la clasificación preliminar descrita antes. La clasificación para microcalcificaciones, en cambio, usa una red neuronal ResNet50, entrenada con una base de datos distinta: CBIS-DDSM. Esto se hizo por la gran complejidad que presentan estas lesiones, y por el hecho del gran desbalance de clases que había en INbreast, habiendo muchas más malignas que benignas.')

    st.subheader('Para más información o para ver el código, consulta el GITHUB del proyecto')







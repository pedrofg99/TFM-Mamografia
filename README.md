# Ayuda y documentación
Este programa permite seleccionar una región de interés (ROI) dentro de una imagen mamográfica. Después, clasifica la imagen recortada según su grado de malignidad, como lesión benigna o lesión maligna. Además, contiene un algoritmo de segmentación que localiza posibles lesiones, con el fin de ayudar al usuario en la selección de la región de interés.

Este programa forma parte del Trabajo Fin de Estudios titulado "Clasificación Automatizada de Lesiones de Mama en Imágenes Médicas Empleando Técnicas Híbridas de Inteligencia Computacional y Aprendizaje Profundo", de autores Pedro Fernández González, Carmen Entremonzaga Melgosa y Ayman El Hichou Chahibou, dirigido por Ivón Oristela Benítez, que forma parte del máster en Ingeniería Matemática y Computación de la UNIR



## Importación de las imágenes y selección de ROI
En primer lugar, deberás indicar la carpeta en la que están tus imágenes mamográficas y seleccionar qué imagen quieres analizar en el menú desplegable. Una vez hecho, deberás seleccionar la región de interés utilizando un rectángulo interactivo

## Segmentación de la imagen
Opcionalmente puedes ayudarte de una segmentación para elegir apropiadamente el ROI. Para ello, selecciona una opción en la selectbox titulada "¿Quieres ayudarte de una segmentación?" Esto mostrará la imagen con la máscara de segmentación superpuesta, en color verde. Se puede hacer zoom libremente a esta imagen.

El algoritmo de segmentación con clústering primero preprocesa la imagen recortada o ROI para diferenciar mejor posibles lesiones, y luego utiliza un método de clústering sobre las intensidades de los píxeles, para separar los píxeles de la imagen en varios umbrales de intensidad. Se seleccionan los píxeles del umbral de mayor intensidad como candidatos a ser lesiones. Después, la máscara se post-procesa para limpiar ruido, separar regiones y quedarse solo con aquellas con un mejor equilibrio entre tamaño y redondez. Esas regiones se consideran posibles masas tumorales. Nótese que este algoritmo solo puede funcionar correctamente para masas, no para otro tipo de lesiones como podrían ser las microcalcificaciones. Además, no aseguramos que el algoritmo localice correctamente qué es una masa tumoral. Por tanto, esto debe tomarse exclusivamente como una posible ayuda, nunca como decisión final de lo que es tumor en la imagen mamográfica.

El algoritmo de segmentación con red neuronal U-net utiliza una red neuronal especializada para realizar la segmentación. Conviene ejecutar los dos y ver dónde coinciden. De todas formas, repetimos, ambos pueden fallar, por lo que la decisión última sobre qué es lesión en la imagen debe hacerla el usuario.

## ¿Este ROI contiene una masa?
En este programa, se hacen dos clasificaciones distintas: como masa o como microcalcificación. Esto es porque cada tipo de lesión necesita criterios distintos para ser clasificada como benigna o maligna. De esa manera, uno de los algoritmos es entrenado solo con imágenes que contienen lesiones tipo masa, y el otro clasificador se entrena solo con imágenes que contienen lesiones tipo microcalcificaciones. Por ello, antes de clasificar se permite, de forma opcional, hacer una clasificación extra que determine si el ROI contiene una masa o no. Eso puede ayudar a decidir, posteriormente, si hacer una clasificación como masa o como microcalcificación. El algoritmo, en primer lugar, preprocesa la imagen recortada o ROI para hacerla apta para los siguientes pasos. Después, transforma la imagen en un conjunto de datos, o features, relacionados con la distribución de intensidades, los gradientes (cambios de intensidad), y las texturas, entre otros. Ese conjunto de datos se pasa por un modelo de Machine Learning SVM (Support Vector Machine), que predice si la imagen es masa o microcalcificación. El modelo fue entrenado con ROIs de masas extraídos del dataset INbreast. Esos ROI se obtuvieron a partir de las máscaras de segmentación manuales hechas por los radiólogos del Dataset.

## Clasificaciones para masas y para microcalcificaciones
El núcleo del programa son las dos clasificaciones finales: en ambas se clasifican las lesiones según su grado de malignidad, como BENIGNAS o MALIGNAS. BENIGNAS corresponden a aquellas con un bi-rads de INbreast menor que 4, y las MALIGNAS corresponden a un bi-rads mayor o igual a 4. En la selectbox "¿Quieres clasificar esta imagen como masa o como microcalcificación?" se puede seleccionar una de las dos opciones. La clasificación para masas usa el mismo algoritmo que la clasificación preliminar descrita antes. La clasificación para microcalcificaciones, en cambio, usa una red neuronal ResNet50, entrenada con una base de datos distinta: CBIS-DDSM. Esto se hizo por la gran complejidad que presentan estas lesiones, y por el hecho del gran desbalance de clases que había en INbreast, habiendo muchas más malignas que benignas.

# Estructura del repositorio
- En /app están los códigos del programa.
- En /models hay un enlace para descargar los modelos utilizados por los códigos del programa.
- En /notebooks están los notebooks en los que diseñamos los algoritmos y entrenamos los modelos de machine learning.

# Instalación
1. Descarga todos los .py de la carpeta /app en una carpeta de tu ordenador. 
2. Descarga los modelos del enlace de Drive en la carpeta /models y guárdalos en la misma carpeta donde están los .py.
3. Asegúrate de tener todas las dependencias que aparecen en requirements.txt instaladas en tu instalación de python. Revisa también las importaciones que aparecen en todos los .py, por si acaso.
4. En la consola, colócate en la carpeta en la que están los códigos y escribe
   
streamlit run roi_app.py

La aplicación debería abrirse automáticamente en tu navegador.

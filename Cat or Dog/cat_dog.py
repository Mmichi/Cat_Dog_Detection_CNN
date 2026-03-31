import streamlit as st  
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np  

# Modelinizi yükleyin
model = load_model('cat_dog_model.h5')

# Resmi işleme fonksiyonu
def process_image(img):
    img = img.resize((170, 170))  # Modelin beklediği boyuta getir 
    img = np.array(img)
    img = img / 255.0  # Normalize et
    img = np.expand_dims(img, axis=0)  # Ekstra boyut ekle
    return img

# Uygulama başlığı
st.markdown('<h1>Kedi mi Köpek mi Sınıflandırma 😾&🐶 </h1>', unsafe_allow_html=True)
st.write('Resim seç, model resminin kedi mi yoksa köpek mi olduğunu tahmin etsin!')

# Dosya yükleyici
file = st.file_uploader('Bir resim yükle', type=['jpg', 'jpeg', 'png'])

if file is not None:  # Resim yüklendiğinde
    img = Image.open(file)
    st.image(img, caption='Yüklenen Resim', use_container_width=True)  # Resmi görüntüle

    image = process_image(img)  # Resmi işle
    try:
        prediction = model.predict(image)  # Modelden tahmin al

        # En yüksek tahmin edilen sınıfı bul
        predicted_class = np.argmax(prediction)

        # Sınıf isimleri
        class_names = ['Cat', 'Dog']
        st.write("Tahmin Edilen Sınıf:", class_names[predicted_class])
    except Exception as e:
        st.write("Tahmin yapılırken bir hata oluştu:", e)  # Hata mesajını göster
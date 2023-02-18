import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np

st.set_option('deprecation.showfileUploaderEncoding', False)
@st.cache(allow_output_mutation=True)

# Loading our pre-trained model
def load():
    model = load_model('cotton_pred_MNv2.h5')
    return model

# model = load_model('cotton_pred_MNv2.h5')

model = load()

# Defining the image size the model expects
img_size = (224, 224)

# Defining the class labels
class_names = ['diseased cotton leaf',
               'diseased cotton plant',
               'fresh cotton leaf',
               'fresh cotton plant']

# Define the Streamlit app
def app():
    st.title('Cotton Plant Disease Predictor 🌿')
    st.write('Upload an image of your cotton plant or leaf, and the AI model will predict if its healthy or diseased.')
    st.write('''If you're here to just try it out, then you can download an image from [Test Dataset](https://drive.google.com/drive/folders/1NNeuZrVMs5ckQx4TBmoaSlkpgTAuDstD?usp=share_link) and upload it here.''')

    # Allow the user to upload an image
    uploaded_file = st.file_uploader('', type=['jpg', 'jpeg', 'png'])

    # If the user has uploaded an image
    if uploaded_file is not None:
        # Load the image and preprocess it
        image = load_img(uploaded_file, target_size=img_size)
        image = img_to_array(image)
        image = preprocess_input(image)
        image = np.expand_dims(image, axis=0)

        # Get the model's prediction for the image
        prediction = model.predict(image)
        predicted_class = class_names[np.argmax(prediction)]

        # Display the prediction to the user
        t = 'This is a '+ predicted_class
        if np.argmax(prediction) < 2:
            st.error(t)
        else:
            st.success(t)
    else:
        st.text('Upload an image file')


# Run the Streamlit app
if __name__ == '__main__':
    app()
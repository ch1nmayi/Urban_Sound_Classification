import streamlit as st
from keras.models import load_model
import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the model
loaded_model = load_model('my_model.h5')

# List of class names corresponding to class indices
class_names = [
    "Air_conditioner", "car_horn", "children_playing", "dog_bark", "drilling",
    "Engine_idling", "gun_shot", "jackhammer", "siren", "street_music"
]

# Function to process a .wav file and make predictions
def predict_class(file_path, class_names):
    x, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
    mfccs = np.mean(librosa.feature.mfcc(y=x, sr=sample_rate, n_mfcc=40).T, axis=0)
    feature = mfccs.reshape(1, -1)
    predicted_probabilities = loaded_model.predict(feature)
    predicted_class_index = np.argmax(predicted_probabilities, axis=1)
    predicted_class_name = class_names[predicted_class_index[0]]
    return predicted_class_name

def main():
    st.title("Urban Sound Classifier")
    st.sidebar.title("Upload Sound")

    uploaded_file = st.sidebar.file_uploader("Choose a sound file", type=["wav"])

    if uploaded_file is not None:
        file_details = {"FileName": uploaded_file.name, "FileType": uploaded_file.type}
        st.sidebar.write(file_details)

        predicted_class = predict_class(uploaded_file, class_names)
        st.success(f"Predicted class: {predicted_class}")

        # Create or load DataFrame to keep track of counts
        if 'class_counts' not in st.session_state:
            st.session_state.class_counts = {class_name: 0 for class_name in class_names}

        st.session_state.class_counts[predicted_class] += 1

        # Display bar graph of class counts
        st.header("Bar Graph - Class Counts")
        st.markdown("This bar graph shows the counts of predicted classes.")
        counts_df = pd.DataFrame.from_dict(st.session_state.class_counts, orient='index', columns=['Count'])
        counts_df = counts_df.sort_values(by='Count', ascending=False)
        st.bar_chart(counts_df)

        # Reset button
        if st.sidebar.button("Reset Counts"):
            st.session_state.class_counts = {class_name: 0 for class_name in class_names}

if __name__ == "__main__":
    main()

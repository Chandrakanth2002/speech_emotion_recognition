import numpy as np
import librosa
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.utils import to_categorical

# Define the emotions
emotions = {
    'neutral': 0,
    'calm': 1,
    'happy': 2,
    'sad': 3,
    'angry': 4,
    'fearful': 5,
    'disgust': 6,
    'surprised': 7
}

# Load and preprocess the audio data
def preprocess_audio(file_path):
    # Load the audio file
    audio, _ = librosa.load(file_path, sr=22050)
    
    # Extract audio features using Mel-frequency cepstral coefficients (MFCC)
    mfcc = librosa.feature.mfcc(audio, sr=22050, n_mfcc=13)
    
    # Pad or truncate the MFCC features to a fixed length of 100 frames
    max_frames = 100
    pad_width = max_frames - mfcc.shape[1]
    mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
    
    return mfcc

# Load the pre-trained model
model = Sequential()
model.add(LSTM(128, return_sequences=True, input_shape=(13, 100)))
model.add(Dropout(0.5))
model.add(LSTM(128))
model.add(Dropout(0.5))
model.add(Dense(8, activation='softmax'))
model.load_weights('emotion_model.h5')

# Recognize speech emotion
def recognize_emotion(file_path):
    # Preprocess the audio
    mfcc = preprocess_audio(file_path)
    
    # Reshape the MFCC features to match the model input shape
    mfcc = np.reshape(mfcc, (1, 13, 100))
    
    # Predict the emotion using the pre-trained model
    predictions = model.predict(mfcc)
    
    # Get the predicted emotions for each prediction
    emotion_predictions=predictions[0]
    # Get the index of the predicted emotion
    predicted_emotion_index = np.argmax(emotion_predictions)

    # Get the predicted emotion label
    predicted_emotion = list(emotions.keys())[list(emotions.values()).index(predicted_emotion_index)]

return predicted_emotion

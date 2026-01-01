# Emotion_Detection_Recommendation.ipynb
!pip install deepface opencv-python-headless textblob pandas
import pandas as pd

data = {
    "Emotion": [
        "Happy","Sad","Angry","Fear","Surprise",
        "Disgust","Neutral","Love","Lonely","Relaxed",
        "Motivated","Depressed","Excited","Calm",
        "Confident","Bored","Hopeful","Grateful",
        "Anxious","Proud"
    ],
    "Song": [
        "Happy – Pharrell Williams",
        "Someone Like You – Adele",
        "Believer – Imagine Dragons",
        "Demons – Imagine Dragons",
        "Uptown Funk – Bruno Mars",
        "Bad Guy – Billie Eilish",
        "Let It Be – Beatles",
        "Perfect – Ed Sheeran",
        "Fix You – Coldplay",
        "Weightless – Marconi Union",
        "Eye of the Tiger – Survivor",
        "1-800-273-8255 – Logic",
        "Can't Stop the Feeling – JT",
        "River – Leon Bridges",
        "Stronger – Kanye West",
        "Stressed Out – Twenty One Pilots",
        "Hall of Fame – The Script",
        "Thank You – Dido",
        "Breathin – Ariana Grande",
        "We Are The Champions – Queen"
    ],
    "Quote": [
        "Happiness is a choice.",
        "Every storm runs out of rain.",
        "Control your anger before it controls you.",
        "Fear is temporary, regret is forever.",
        "Life is full of surprises.",
        "Negativity harms the soul.",
        "Peace begins with acceptance.",
        "Love is the strongest emotion.",
        "You are never truly alone.",
        "Calm mind brings inner strength.",
        "Push yourself, no one else will.",
        "This feeling will pass.",
        "Energy flows where attention goes.",
        "Silence heals the soul.",
        "Believe in your abilities.",
        "Boredom is unused creativity.",
        "Hope is stronger than fear.",
        "Gratitude turns what we have into enough.",
        "Anxiety does not define you.",
        "Be proud of how far you’ve come."
    ]
}

df = pd.DataFrame(data)
df.to_csv("emotions_data.csv", index=False)

# The previous command was incomplete, so let's complete it and remove the duplicate import

emotion_df = pd.read_csv("emotions_data.csv")

from textblob import TextBlob

def detect_text_emotion(text):
    polarity = TextBlob(text).sentiment.polarity

    if polarity > 0.5:
        return "Happy"
    elif polarity > 0.1:
        return "Hopeful"
    elif polarity < -0.5:
        return "Depressed"
    elif polarity < -0.1:
        return "Sad"
    else:
        return "Neutral"

from deepface import DeepFace
import cv2

def detect_face_emotion(image_path):
    result = DeepFace.analyze(
        img_path=image_path,
        actions=['emotion'],
        enforce_detection=False
    )
    return result[0]['dominant_emotion']

from google.colab import files

def recommend(emotion):
    row = emotion_df[emotion_df["Emotion"].str.lower() == emotion.lower()]
    if not row.empty:
        return row.iloc[0]["Song"], row.iloc[0]["Quote"]
    else:
        return "No song found", "No quote found"

# User interaction for text emotion
text = input("Enter your feeling text: ")
emotion_from_text = detect_text_emotion(text)

song_text, quote_text = recommend(emotion_from_text)

print("Detected Emotion (Text):", emotion_from_text)
print("Recommended Song (Text):", song_text)
print("Quote (Text):", quote_text)

# User interaction for face emotion
print("Please upload an image for face emotion detection.")
uploaded = files.upload()

if uploaded:
    image_path = list(uploaded.keys())[0]
    emotion_from_face = detect_face_emotion(image_path)
    song_face, quote_face = recommend(emotion_from_face)

    print("Detected Emotion (Face):", emotion_from_face)
    print("Recommended Song (Face):", song_face)
    print("Quote (Face):", quote_face)
else:
    print("No image uploaded for face emotion detection.")

from flask import Flask, render_template, request, jsonify
import os
import base64
import random

from recommender import get_recommendations

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'

# Initialize models
print("Starting MoodMate...")

# Simple text emotion detection
def predict_emotion_from_text(text):
    text_lower = text.lower()
    
    if any(word in text_lower for word in ['happy', 'joy', 'good', 'great', 'excited', 'awesome', 'wonderful']):
        return 'Happy'
    elif any(word in text_lower for word in ['sad', 'unhappy', 'depressed', 'cry', 'upset', 'miserable']):
        return 'Sad'
    elif any(word in text_lower for word in ['angry', 'mad', 'furious', 'hate', 'annoyed', 'irritated']):
        return 'Angry'
    elif any(word in text_lower for word in ['fear', 'scared', 'afraid', 'nervous', 'anxious', 'terrified']):
        return 'Fearful'
    elif any(word in text_lower for word in ['surprise', 'shock', 'wow', 'amazed', 'astonished']):
        return 'Surprised'
    elif any(word in text_lower for word in ['disgust', 'hate', 'ugly', 'revolting', 'disgusting']):
        return 'Disgusted'
    else:
        return 'Neutral'

# Fallback image emotion detection
def predict_emotion_from_image(image_bytes):
    emotions = ['Happy', 'Sad', 'Angry', 'Neutral', 'Surprised', 'Fearful', 'Disgusted']
    return random.choice(emotions)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/detect_emotion', methods=['POST'])
def detect_emotion():
    try:
        if 'image' in request.files:
            file = request.files['image']
            if file.filename == '':
                return jsonify({'error': 'No file selected', 'success': False})
            img_bytes = file.read()
            emotion = predict_emotion_from_image(img_bytes)
        elif 'text' in request.form:
            text = request.form['text']
            if not text.strip():
                return jsonify({'error': 'No text provided', 'success': False})
            emotion = predict_emotion_from_text(text)
        else:
            return jsonify({'error': 'Please provide image or text', 'success': False})
        
        # Use your exact get_recommendations function
        recommendations_data = get_recommendations([emotion], 8)
        
        # Convert to required format
        recommendations = []
        for _, row in recommendations_data.iterrows():
            recommendations.append({
                'title': row['name'],
                'artist': row['artist'],
                'mood': emotion
            })
        
        return jsonify({
            'emotion': emotion,
            'recommendations': recommendations,
            'success': True
        })
        
    except Exception as e:
        return jsonify({'error': str(e), 'success': False})

@app.route('/webcam', methods=['POST'])
def webcam_capture():
    # Simple webcam - always return random emotion
    emotions = ['Happy', 'Sad', 'Angry', 'Neutral', 'Surprised']
    emotion = random.choice(emotions)
    
    recommendations_data = get_recommendations([emotion], 8)
    recommendations = []
    for _, row in recommendations_data.iterrows():
        recommendations.append({
            'title': row['name'],
            'artist': row['artist'],
            'mood': emotion
        })
    
    return jsonify({
        'emotion': emotion,
        'recommendations': recommendations,
        'success': True
    })

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True, host='127.0.0.1', port=5000)
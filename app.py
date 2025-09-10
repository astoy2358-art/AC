from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import Levenshtein
import tensorflow as tf
from tensorflow import keras
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import os

# Download NLTK data
nltk.download('punkt')

app = Flask(__name__)

# Global variables for models and data
word_freq_df = None
seq2seq_model = None
char_to_idx = None
idx_to_char = None
max_seq_length = 20

def load_data():
    global word_freq_df, char_to_idx, idx_to_char
    
    # Load word corpus
    word_freq_df = pd.read_csv('word_corpus.csv')
    word_freq_df['word'] = word_freq_df['word'].astype(str)
    word_freq_df['frequency'] = word_freq_df['frequency'].astype(int)
    
    # Create character mappings
    chars = set()
    for word in word_freq_df['word']:
        chars.update(list(word))
    
    chars = sorted(list(chars))
    char_to_idx = {char: idx for idx, char in enumerate(chars)}
    idx_to_char = {idx: char for idx, char in enumerate(chars)}
    
    # Generate visualization
    generate_visualization()

def generate_visualization():
    # Get top 25 words by frequency
    top_words = word_freq_df.nlargest(25, 'frequency')
    
    # Create horizontal bar chart
    plt.figure(figsize=(10, 8))
    plt.barh(top_words['word'], top_words['frequency'])
    plt.xlabel('Frequency')
    plt.ylabel('Words')
    plt.title('Top 25 Most Frequent Words')
    plt.tight_layout()
    
    # Save the visualization
    if not os.path.exists('static/images'):
        os.makedirs('static/images')
    plt.savefig('static/images/word_frequency.png')
    plt.close()

def load_model():
    global seq2seq_model
    seq2seq_model = tf.keras.models.load_model('models/seq2seq_model.h5')

def tier1_correct(word):
    # If word exists in corpus, return as is
    if word in word_freq_df['word'].values:
        return word
    
    # Get candidates with edit distance 1 or 2
    candidates = []
    for _, row in word_freq_df.iterrows():
        dist = Levenshtein.distance(word, row['word'])
        if dist <= 2:
            candidates.append((row['word'], row['frequency'], dist))
    
    if not candidates:
        return None
    
    # Sort by distance then frequency
    candidates.sort(key=lambda x: (x[2], -x[1]))
    return candidates[0][0]

def tier2_correct(word):
    # Preprocess word for model
    word_seq = np.zeros((1, max_seq_length, len(char_to_idx)))
    for i, char in enumerate(word[:max_seq_length]):
        if char in char_to_idx:
            word_seq[0, i, char_to_idx[char]] = 1
    
    # Predict
    prediction = seq2seq_model.predict(word_seq, verbose=0)
    
    # Convert prediction to word
    corrected_word = ''
    for i in range(max_seq_length):
        char_idx = np.argmax(prediction[0, i, :])
        if char_idx == 0:  # Padding token
            break
        corrected_word += idx_to_char[char_idx]
    
    return corrected_word

def correct_text(text):
    sentences = sent_tokenize(text)
    corrected_sentences = []
    
    for sentence in sentences:
        words = word_tokenize(sentence)
        corrected_words = []
        
        for word in words:
            # Skip non-alphabetic tokens
            if not word.isalpha():
                corrected_words.append(word)
                continue
            
            # Try Tier 1 correction
            correction = tier1_correct(word.lower())
            
            # If Tier 1 fails, try Tier 2
            if correction is None:
                correction = tier2_correct(word.lower())
            
            # Preserve case
            if word.istitle():
                correction = correction.title()
            elif word.isupper():
                correction = correction.upper()
                
            corrected_words.append(correction)
        
        corrected_sentences.append(' '.join(corrected_words))
    
    return ' '.join(corrected_sentences)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/correct', methods=['POST'])
def api_correct():
    try:
        data = request.get_json()
        text = data.get('text', '')
        
        if not text:
            return jsonify({'corrected_text': ''})
        
        corrected_text = correct_text(text)
        return jsonify({'corrected_text': corrected_text})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    load_data()
    load_model()
    app.run(debug=True)

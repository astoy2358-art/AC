import pandas as pd
import numpy as np
import random

def introduce_errors(word, error_rate=0.3):
    """Introduce realistic errors to a word"""
    if len(word) <= 1 or random.random() > error_rate:
        return word
    
    operations = ['delete', 'insert', 'substitute', 'transpose']
    operation = random.choice(operations)
    
    pos = random.randint(0, len(word)-1)
    
    if operation == 'delete' and len(word) > 1:
        # Delete a character
        return word[:pos] + word[pos+1:]
    
    elif operation == 'insert':
        # Insert a random character
        char = random.choice('abcdefghijklmnopqrstuvwxyz')
        return word[:pos] + char + word[pos:]
    
    elif operation == 'substitute':
        # Substitute a character
        char = random.choice('abcdefghijklmnopqrstuvwxyz')
        return word[:pos] + char + word[pos+1:]
    
    elif operation == 'transpose' and len(word) > pos+1:
        # Transpose adjacent characters
        return word[:pos] + word[pos+1] + word[pos] + word[pos+2:]
    
    return word

def generate_training_data():
    # Load word corpus
    df = pd.read_csv('word_corpus.csv')
    
    # Generate training data
    training_data = []
    
    for _, row in df.iterrows():
        word = row['word']
        # Generate multiple error variations for each word
        for _ in range(3):
            misspelled = introduce_errors(word)
            if misspelled != word:
                training_data.append((misspelled, word))
    
    # Create DataFrame and save
    training_df = pd.DataFrame(training_data, columns=['input', 'output'])
    training_df.to_csv('training_data.csv', index=False)
    print(f"Generated {len(training_df)} training samples")

if __name__ == '__main__':
    generate_training_data()

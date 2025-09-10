import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding, Attention
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

def create_character_mappings():
    # Create character vocabulary
    chars = set()
    training_df = pd.read_csv('training_data.csv')
    
    for word in training_df['input']:
        chars.update(list(str(word)))
    for word in training_df['output']:
        chars.update(list(str(word)))
    
    chars = sorted(list(chars))
    char_to_idx = {char: idx+1 for idx, char in enumerate(chars)}  # 0 is reserved for padding
    idx_to_char = {idx+1: char for idx, char in enumerate(chars)}
    
    return char_to_idx, idx_to_char, len(chars) + 1  # +1 for padding

def prepare_data(char_to_idx, max_seq_length=20):
    training_df = pd.read_csv('training_data.csv')
    
    # Convert words to sequences
    X = []
    y = []
    
    for _, row in training_df.iterrows():
        input_seq = [char_to_idx.get(c, 0) for c in str(row['input'])]
        output_seq = [char_to_idx.get(c, 0) for c in str(row['output'])]
        
        # Pad sequences
        input_seq = pad_sequences([input_seq], maxlen=max_seq_length, padding='post')[0]
        output_seq = pad_sequences([output_seq], maxlen=max_seq_length, padding='post')[0]
        
        X.append(input_seq)
        y.append(output_seq)
    
    X = np.array(X)
    y = np.array(y)
    
    # Convert y to categorical
    y_categorical = to_categorical(y, num_classes=len(char_to_idx)+1)
    
    return X, y_categorical

def build_seq2seq_model(vocab_size, max_seq_length=20, embedding_dim=50, latent_dim=128):
    # Encoder
    encoder_inputs = Input(shape=(max_seq_length,))
    encoder_embedding = Embedding(vocab_size, embedding_dim, mask_zero=True)(encoder_inputs)
    encoder_lstm = LSTM(latent_dim, return_state=True)
    encoder_outputs, state_h, state_c = encoder_lstm(encoder_embedding)
    encoder_states = [state_h, state_c]
    
    # Decoder
    decoder_inputs = Input(shape=(max_seq_length,))
    decoder_embedding = Embedding(vocab_size, embedding_dim, mask_zero=True)(decoder_inputs)
    decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
    
    # Attention mechanism
    attention = Attention()([decoder_outputs, encoder_outputs])
    decoder_combined = keras.layers.Concatenate()([decoder_outputs, attention])
    
    # Dense layer
    decoder_dense = Dense(vocab_size, activation='softmax')
    output = decoder_dense(decoder_combined)
    
    # Model
    model = Model([encoder_inputs, decoder_inputs], output)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

def train_model():
    char_to_idx, idx_to_char, vocab_size = create_character_mappings()
    X, y = prepare_data(char_to_idx)
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create decoder input (shifted right)
    decoder_input_data = np.zeros_like(X_train)
    decoder_input_data[:, 1:] = X_train[:, :-1]
    decoder_input_data[:, 0] = 0  # Start token
    
    # Build and train model
    model = build_seq2seq_model(vocab_size)
    model.fit(
        [X_train, decoder_input_data], y_train,
        batch_size=64,
        epochs=10,
        validation_split=0.2
    )
    
    # Save model
    model.save('models/seq2seq_model.h5')
    print("Model trained and saved successfully")

if __name__ == '__main__':
    train_model()

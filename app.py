# app.py - Flask Backend for Image Captioning Model
from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
from torchvision import transforms
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from PIL import Image
import io
import base64
import json
import os
import re
from collections import Counter
from tqdm import tqdm
from torchvision import models
import torch.nn.functional as F
from vocabulary import Vocabulary
from config import Config
from encodercnn import EncoderCNN
from model import ImageCaptioningModel
from attention import Attention

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

# Global variables for model components
encoder = None
decoder = None
vocab = None
transform = None
device = None


# Configuration Class
# ============================================



config = Config()

# Create necessary directories
#os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)



# ============================================
# Decoder with Attention
# ============================================






def beam_search(encoder, decoder, image, vocab, beam_size=3):
    """
    Generate caption using beam search
    """
    k = beam_size
    vocab_size = len(vocab)
    
    # Encode image
    encoder_out = encoder(image)  # (1, enc_image_size, enc_image_size, encoder_dim)
    enc_image_size = encoder_out.size(1)
    encoder_dim = encoder_out.size(3)
    
    # Flatten encoding
    encoder_out = encoder_out.view(1, -1, encoder_dim)  # (1, num_pixels, encoder_dim)
    num_pixels = encoder_out.size(1)
    
    # Expand encoder_out for beam search
    encoder_out = encoder_out.expand(k, num_pixels, encoder_dim)  # (k, num_pixels, encoder_dim)
    
    # Initialize beam search
    k_prev_words = torch.LongTensor([[vocab.word2idx['<START>']]] * k).to(device)  # (k, 1)
    seqs = k_prev_words  # (k, 1)
    top_k_scores = torch.zeros(k, 1).to(device)  # (k, 1)
    
    # Lists to store completed sequences and scores
    complete_seqs = []
    complete_seqs_scores = []
    
    # Initialize hidden states
    h, c = decoder.init_hidden_state(encoder_out)
    
    # Start decoding
    step = 1
    while True:
        embeddings = decoder.embedding(k_prev_words).squeeze(1)  # (s, embed_dim)
        
        attention_weighted_encoding, alpha = decoder.attention(encoder_out, h)
        gate = decoder.sigmoid(decoder.f_beta(h))
        attention_weighted_encoding = gate * attention_weighted_encoding
        
        h, c = decoder.decode_step(
            torch.cat([embeddings, attention_weighted_encoding], dim=1),
            (h, c)
        )
        
        scores = decoder.fc(h)  # (s, vocab_size)
        scores = F.log_softmax(scores, dim=1)
        
        # Add to previous scores
        scores = top_k_scores.expand_as(scores) + scores  # (s, vocab_size)
        
        # For the first step, all k points will have the same scores (since same k previous words)
        if step == 1:
            top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)  # (s)
        else:
            # Unroll and find top scores, and their unrolled indices
            top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)  # (s)
            
        # Convert unrolled indices to actual indices of scores
        prev_word_inds = top_k_words // vocab_size  # (s)
        next_word_inds = top_k_words % vocab_size  # (s)
        
        # Add new words to sequences
        seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)
        
        # Which sequences are incomplete (didn't reach <end>)?
        incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) 
                          if next_word != vocab.word2idx['<END>']]
        complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))
        
        # Set aside complete sequences
        if len(complete_inds) > 0:
            complete_seqs.extend(seqs[complete_inds].tolist())
            complete_seqs_scores.extend(top_k_scores[complete_inds])
            
        k -= len(complete_inds)  # reduce beam length accordingly
        
        # Proceed with incomplete sequences
        if k == 0:
            break
            
        seqs = seqs[incomplete_inds]
        h = h[prev_word_inds[incomplete_inds]]
        c = c[prev_word_inds[incomplete_inds]]
        encoder_out = encoder_out[prev_word_inds[incomplete_inds]]
        top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
        k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)
        
        # Break if we've been decoding too long
        if step > 50:
            break
        step += 1
        
    # Select best sequence
    if len(complete_seqs_scores) > 0:
        i = complete_seqs_scores.index(max(complete_seqs_scores))
        seq = complete_seqs[i]
    else:
        seq = seqs[0].tolist()
        
    return seq

def load_model(model_path=r'C:\Users\User\Downloads\best_model_clean.pth'):
    """Load the trained model"""
    global encoder, decoder, vocab, transform, device
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        # Try loading clean checkpoint first
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
        
        # Create vocabulary from saved dictionaries
        vocab = Vocabulary()
        vocab.word2idx = checkpoint['word2idx']
        vocab.idx2word = checkpoint['idx2word']
        
        # Create config
        config = Config(**checkpoint['config'])
        
    except Exception as e:
        print(f"Failed to load clean checkpoint, trying with weights_only=False: {e}")
        # Allow custom classes for loading
        torch.serialization.add_safe_globals([Vocabulary, Config])
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        vocab = checkpoint['vocab']
        config = checkpoint['config']
    
    # Initialize model
    model = ImageCaptioningModel(len(vocab), config)
    model.to(device)
    
    # Load state dictionaries
    model.encoder.load_state_dict(checkpoint['encoder_state_dict'])
    model.decoder.load_state_dict(checkpoint['decoder_state_dict'])
    
    # Set to evaluation mode
    model.encoder.eval()
    model.decoder.eval()
    
    # Store components globally
    encoder = model.encoder
    decoder = model.decoder
    
    # Define image preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    print(f"✅ Model loaded successfully on {device}")
    return True

def generate_caption(image_tensor, max_len=20):
    """Generate caption for an image"""
    with torch.no_grad():
        # Get features from encoder
        features = encoder(image_tensor.unsqueeze(0))  # (1, 14, 14, 2048)
        features = features.view(features.size(0), -1, features.size(-1))  # (1, 196, 2048)

        # Initialize hidden state
        h, c = decoder.init_hidden_state(features)

        sampled_ids = []
        # Start with the <START> token
        inputs = torch.LongTensor([[vocab.word2idx.get('<START>', vocab.word2idx.get('<start>', 0))]]).to(device)

        for _ in range(max_len):
            embeddings = decoder.embedding(inputs).squeeze(1)  # (1, embed_dim)

            # Apply attention
            attention_weighted_encoding, alpha = decoder.attention(features, h)
            gate = decoder.sigmoid(decoder.f_beta(h))
            attention_weighted_encoding = gate * attention_weighted_encoding

            # LSTM decode step
            h, c = decoder.decode_step(
                torch.cat([embeddings, attention_weighted_encoding], dim=1),
                (h, c)
            )

            # Predict next word
            outputs = decoder.fc(h)  # (1, vocab_size)
            predicted = outputs.argmax(1)  # (1)

            sampled_ids.append(predicted.item())

            # Stop if <END> token is predicted
            end_token = vocab.word2idx.get('<END>', vocab.word2idx.get('<end>', 1))
            if predicted.item() == end_token:
                break

            # Use predicted word as input for next step
            inputs = predicted.unsqueeze(0)  # (1, 1)

    # Convert token IDs to words
    words = []
    for idx in sampled_ids:
        word = vocab.idx2word.get(idx, '<UNK>')
        if word not in ['<END>', '<end>']:
            words.append(word)
    
    return " ".join(words)

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': encoder is not None and decoder is not None
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Generate caption for uploaded image"""
    try:
        if encoder is None or decoder is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        # Check if image is provided
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        image_file = request.files['image']
        if image_file.filename == '':
            return jsonify({'error': 'No image selected'}), 400
        
        # Process image
        image = Image.open(image_file.stream).convert('RGB')
        image_tensor = transform(image).to(device)
        
        # Generate caption
        caption = generate_caption(image_tensor)
        
        return jsonify({
            'caption': caption,
            'status': 'success'
        })
        
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/predict_base64', methods=['POST'])
def predict_base64():
    """Generate caption for base64 encoded image"""
    try:
        if encoder is None or decoder is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        data = request.get_json()
        if 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400
        
        # Decode base64 image
        image_data = base64.b64decode(data['image'].split(',')[1] if ',' in data['image'] else data['image'])
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        
        # Process image
        image_tensor = transform(image).to(device)
        
        # Generate caption
        caption = generate_caption(image_tensor)
        
        return jsonify({
            'caption': caption,
            'status': 'success'
        })
        
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/model_info', methods=['GET'])
def model_info():
    """Get model information"""
    if vocab is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    return jsonify({
        'vocab_size': len(vocab.word2idx),
        'device': str(device),
        'model_loaded': True
    })

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Image Captioning API Server')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model checkpoint')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host address')
    parser.add_argument('--port', type=int, default=5000, help='Port number')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()
    
    # Load model
    if load_model(args.model_path):
        print(f" Starting server on {args.host}:{args.port}")
        app.run(host=args.host, port=args.port, debug=args.debug)
    else:
        print("Failed to load model. Server not started.")
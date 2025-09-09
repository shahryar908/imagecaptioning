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

class Vocabulary:
    """Handles word-to-index mappings and text processing"""
    
    def __init__(self, freq_threshold=5):
        self.freq_threshold = freq_threshold
        self.word2idx = {
            "<PAD>": 0,
            "<START>": 1, 
            "<END>": 2,
            "<UNK>": 3
        }
        self.idx2word = {v: k for k, v in self.word2idx.items()}
        self.word_freq = Counter()
        
    def build_vocabulary(self, caption_list):
        """Build vocabulary from list of captions"""
        # Count word frequencies
        for caption in tqdm(caption_list, desc="Building vocabulary"):
            tokens = self.tokenize(caption)
            self.word_freq.update(tokens)
        
        # Add words that meet frequency threshold
        words = [word for word, count in self.word_freq.items() 
                if count >= self.freq_threshold]
        
        for idx, word in enumerate(words, start=len(self.word2idx)):
            self.word2idx[word] = idx
            self.idx2word[idx] = word
            
        print(f"Vocabulary size: {len(self.word2idx)} words")
        print(f"Most common words: {self.word_freq.most_common(10)}")
        
    def tokenize(self, text):
        """Tokenize and clean text"""
        # Convert to lowercase
        text = text.lower().strip()
        # Remove punctuation except apostrophes
        text = re.sub(r'[^\w\s\']', '', text)
        # Tokenize
        tokens = text.split()
        return tokens
    
    def encode(self, text):
        """Convert text to list of indices"""
        tokens = self.tokenize(text)
        indices = []
        for token in tokens:
            if token in self.word2idx:
                indices.append(self.word2idx[token])
            else:
                indices.append(self.word2idx["<UNK>"])
        return indices
    
    def decode(self, indices):
        """Convert indices back to text"""
        words = []
        for idx in indices:
            if idx == self.word2idx["<END>"]:
                break
            if idx not in [self.word2idx["<PAD>"], self.word2idx["<START>"]]:
                words.append(self.idx2word.get(idx, "<UNK>"))
        return ' '.join(words)
    
    def __len__(self):
        return len(self.word2idx)
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
from encodercnn import EncoderCNN
from decoderwithattention import DecoderWithAttention
from config import device, config

class ImageCaptioningModel(nn.Module):
    """Complete image captioning model with encoder and decoder"""

    def __init__(self, vocab_size, config):
        super(ImageCaptioningModel, self).__init__()
        self.config = config
        self.vocab_size = vocab_size

        # Initialize encoder and decoder
        self.encoder = EncoderCNN().to(device)
        self.decoder = DecoderWithAttention(
            attention_dim=config.ATTENTION_DIM,
            embed_dim=config.EMBED_SIZE,
            decoder_dim=config.DECODER_DIM,
            vocab_size=vocab_size,
            dropout=config.DROPOUT
        ).to(device)

        # Initialize the encoder with fine-tuning enabled
        self.encoder.fine_tune(True)

        # Optimizers
        self.encoder_optimizer = torch.optim.Adam(
            params=filter(lambda p: p.requires_grad, self.encoder.parameters()),
            lr=config.LEARNING_RATE
        )

        self.decoder_optimizer = torch.optim.Adam(
            params=filter(lambda p: p.requires_grad, self.decoder.parameters()),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )

        # Loss function
        self.criterion = nn.CrossEntropyLoss().to(device)

    def train_step(self, images, captions, caption_lengths):
        """One training step"""
        # Move to device
        images = images.to(device)
        captions = captions.to(device)
        caption_lengths = caption_lengths.to(device)

        # Forward pass through encoder
        encoder_out = self.encoder(images)

        # Forward pass through decoder
        predictions, caps_sorted, decode_lengths, alphas, sort_ind = self.decoder(
            encoder_out, captions, caption_lengths
        )

        # Since we decoded starting with <start>, the targets are all words after <start>
        targets = caps_sorted[:, 1:]

        # Remove timesteps that we didn't decode at, or are pads
        predictions = pack_padded_sequence(predictions, decode_lengths, batch_first=True).data
        targets = pack_padded_sequence(targets, decode_lengths, batch_first=True).data

        # Calculate loss
        loss = self.criterion(predictions, targets)

        # Add doubly stochastic attention regularization
        loss += 1.0 * ((1. - alphas.sum(dim=1)) ** 2).mean()

        # Back prop
        self.decoder_optimizer.zero_grad()
        if self.encoder_optimizer is not None:
            self.encoder_optimizer.zero_grad()

        loss.backward()

        # Clip gradients
        torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), config.GRAD_CLIP)
        if self.encoder_optimizer is not None:
            torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), config.GRAD_CLIP)

        # Update weights
        self.decoder_optimizer.step()
        if self.encoder_optimizer is not None:
            self.encoder_optimizer.step()

        return loss.item()

    def validate(self, val_loader):
        """Validate the model"""
        self.encoder.eval()
        self.decoder.eval()

        total_loss = 0
        with torch.no_grad():
            for images, captions, caption_lengths in tqdm(val_loader, desc='Validating'):
                images = images.to(device)
                captions = captions.to(device)
                caption_lengths = caption_lengths.to(device)

                # Forward pass
                encoder_out = self.encoder(images)
                predictions, caps_sorted, decode_lengths, alphas, sort_ind = self.decoder(
                    encoder_out, captions, caption_lengths
                )

                # Calculate loss
                targets = caps_sorted[:, 1:]
                predictions = pack_padded_sequence(predictions, decode_lengths, batch_first=True).data
                targets = pack_padded_sequence(targets, decode_lengths, batch_first=True).data

                loss = self.criterion(predictions, targets)
                loss += 1.0 * ((1. - alphas.sum(dim=1)) ** 2).mean()

                total_loss += loss.item()

        self.encoder.train()
        self.decoder.train()

        return total_loss / len(val_loader)
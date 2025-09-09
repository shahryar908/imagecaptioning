class Config:
    """Configuration class for all hyperparameters"""
    # Paths - Update these based on your setup
    DATASET_PATH = '/content'
    IMAGES_DIR = '/content/Images'
    CAPTIONS_FILE = '/content/captions.txt'
    
    # Model architecture
    ENCODER_DIM = 2048  # ResNet output dimension
    EMBED_SIZE = 512
    ATTENTION_DIM = 512
    DECODER_DIM = 512
    DROPOUT = 0.5
    
    # Training parameters
    BATCH_SIZE = 32
    NUM_EPOCHS = 15
    LEARNING_RATE = 4e-4
    WEIGHT_DECAY = 1e-5
    GRAD_CLIP = 5.0
    PRINT_FREQ = 100
    
    # Data parameters
    MAX_CAPTION_LENGTH = 50
    MIN_WORD_FREQ = 5
    NUM_WORKERS = 2
    
    # Training splits
    TRAIN_SPLIT = 0.8
    VAL_SPLIT = 0.1
    TEST_SPLIT = 0.1
    
    # Regularization
    TEACHER_FORCING_RATIO = 1.0  # Will be decayed during training
    SCHEDULED_SAMPLING_START = 5  # Epoch to start scheduled sampling
    
    # Checkpointing
    CHECKPOINT_DIR = '/content/checkpoints'
    SAVE_EVERY = 5
    
    # Early stopping
    PATIENCE = 7
    MIN_DELTA = 0.0001
    
    # Beam search
    BEAM_SIZE = 3
    
    # Augmentation
    USE_AUGMENTATION = True
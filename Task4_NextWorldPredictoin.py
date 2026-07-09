import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# 1. DATA - Chhota dataset de raha hu, tum bada bhi use kar sakte ho
text = """
Artificial intelligence is transforming the world.
Machine learning is a subset of artificial intelligence.
Deep learning uses neural networks with many layers.
CodSoft internship is great for learning AI.
Python is the best language for artificial intelligence.
"""

# 2. TOKENIZE
tokenizer = Tokenizer()
tokenizer.fit_on_texts([text])
total_words =

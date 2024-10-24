!pip install pandas numpy gensim nltk scikit-learn -q

import pandas as pd
import numpy as np
import gensim
from gensim import corpora
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import WordNetLemmatizer
import nltk

# Download NLTK resources if not already installed
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Sample data
data = [
    "The cat sat on the mat.",
    "Dogs are great pets.",
    "I love to play with my dog.",
    "Cats are independent animals.",
    "My pet loves to sit on my lap.",
    "Animals are fun to watch."
]

# Preprocessing function
def preprocess_text(text):
    # Tokenize
    tokens = word_tokenize(text.lower())
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return tokens

# Preprocess the data
processed_data = [preprocess_text(sentence) for sentence in data]

# Create a dictionary representation of the documents
dictionary = corpora.Dictionary(processed_data)

# Create a corpus: Bag of Words
corpus = [dictionary.doc2bow(text) for text in processed_data]

# Define the LDA model
lda_model = gensim.models.LdaModel(
    corpus, 
    num_topics=2,  # You can adjust this to control the number of topics
    id2word=dictionary, 
    passes=10, 
    random_state=42
)

# Print the topics discovered
topics = lda_model.print_topics(num_words=1000)
print("Identified Topics:")
for topic in topics:
    print(topic)

# For each document, find the most probable topic
print("\nDocument Categorization:")
for idx, sentence in enumerate(data):
    bow = dictionary.doc2bow(preprocess_text(sentence))
    topic_distribution = lda_model[bow]
    print(f"Sentence: '{sentence}' is classified as Topic {max(topic_distribution, key=lambda x: x[1])[0]}")

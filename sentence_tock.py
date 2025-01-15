import nltk
from nltk.tokenize import sent_tokenize
# Download the necessary data (only needed once)
nltk.download('punkt')
# Sampletext
text = "Hello there! How are you doing today? NLP is fascinating."
# Tokenize into sentences
sentences = sent_tokenize(text)
print(sentences)
import string
# Sample text
text = "Hello there! How are you doing today? NLP is fascinating."
# Remove punctuation
clean_text = text.translate(str.maketrans('', '', string.punctuation))
print(clean_text)
import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag
# Download necessary data
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
# Sample text
text = "NLP is fascinating. It enables computers to understand human language."

# Tokenize the text
tokens = word_tokenize(text)

# Perform POS tagging
pos_tags = pos_tag(tokens)
print(pos_tags)

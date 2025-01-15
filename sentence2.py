import nltk
from nltk.tokenize import sent_tokenize
# Download the necessary data (only needed once)
nltk.download('punkt')
# Sampletext
#text = "Hello there! How are you doing today? NLP is fascinating."
# Tokenize into sentences
#sentences = sent_tokenize(text)
#print(sentences)
with open('sentence3.txt', 'r') as file:
    # Read all lines from the file
    content = file.readlines()
    print(content)
    sentences = sent_tokenize(content)
    print(sentences)

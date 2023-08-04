import re
import string
from unidecode import unidecode
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

def textPreprocessingFunc(text):
    # Removing accents
    text = unidecode(text)

    # Strip
    text = text.strip()

    # Removing special characters and numbers
    charsToRemove = string.punctuation + "0123456789"
    text = re.sub(r"["+charsToRemove+"]", "", text)

    # Removing multiple white spaces and tabs
    text = re.sub(" +|\t", " ", text)

    # Uppercase
    text = text.upper()

    return text

def vectorizeText(text, useStemmer = False):
    text = textPreprocessingFunc(text)
    tokens = word_tokenize(text, language = "english", preserve_line = False)
    
    # Removin Stopwords
    stopWords = set(stopwords.words('english'))
    tokens = [token for token in tokens if not token.lower in stopWords]

    if useStemmer:
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(token) for token in tokens]

    return tokens
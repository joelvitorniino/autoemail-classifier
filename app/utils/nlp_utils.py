import re
import unicodedata
import nltk
from nltk.corpus import stopwords
from nltk.stem import RSLPStemmer
from typing import List

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('rslp')

class NLPProcessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('portuguese'))
        self.stemmer = RSLPStemmer()
    
    def remove_accents(self, text: str) -> str:
        return ''.join(c for c in unicodedata.normalize('NFD', text)
                     if unicodedata.category(c) != 'Mn')
    
    def clean_email_headers(self, text: str) -> str:
        headers = [
            r'From:.*?\n', r'To:.*?\n', r'Subject:.*?\n', r'Date:.*?\n',
            r'Cc:.*?\n', r'Bcc:.*?\n', r'Reply-To:.*?\n', r'Return-Path:.*?\n',
            r'Received:.*?\n', r'Message-ID:.*?\n', r'X-.*?:.*?\n',
            r'Content-Type:.*?\n', r'Content-Transfer-Encoding:.*?\n', r'MIME-Version:.*?\n'
        ]
        for h in headers:
            text = re.sub(h, '', text, flags=re.IGNORECASE)
        return text
    
    def tokenize_and_clean(self, text: str) -> List[str]:
        text = self.remove_accents(text.lower())
        text = self.clean_email_headers(text)
        text = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', text)
        text = re.sub(r'http[s]?://\S+', '', text)
        text = re.sub(r'[\w\.-]+@[\w\.-]+\.\w+', '', text)
        text = re.sub(r'\(?\d{2}\)?\s*\d{4,5}-\d{4}', '', text)
        text = re.sub(r'[^\w\s]', ' ', text)
        tokens = nltk.word_tokenize(text)
        return [self.stemmer.stem(token) for token in tokens if token not in self.stop_words and len(token) > 2]
    
    def extract_key_phrases(self, text: str, top_n: int = 5) -> List[str]:
        tokens = self.tokenize_and_clean(text)
        freq_dist = nltk.FreqDist(tokens)
        return [word for word, _ in freq_dist.most_common(top_n)]
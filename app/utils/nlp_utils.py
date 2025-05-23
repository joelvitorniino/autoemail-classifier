import re
import unicodedata
import nltk
from nltk.corpus import stopwords
from nltk.stem import RSLPStemmer
from typing import List

class NLPProcessor:
    """Processes and analyzes Portuguese text with NLP techniques."""
    
    # Regex patterns compiled once at class level
    HEADER_PATTERNS = [
        re.compile(pattern, re.IGNORECASE) for pattern in [
            r'From:.*?\n', r'To:.*?\n', r'Subject:.*?\n', r'Date:.*?\n',
            r'Cc:.*?\n', r'Bcc:.*?\n', r'Reply-To:.*?\n', r'Return-Path:.*?\n',
            r'Received:.*?\n', r'Message-ID:.*?\n', r'X-.*?:.*?\n',
            r'Content-Type:.*?\n', r'Content-Transfer-Encoding:.*?\n', 
            r'MIME-Version:.*?\n'
        ]
    ]
    
    CLEANING_PATTERNS = {
        'control_chars': re.compile(r'[\x00-\x1F\x7F-\x9F]'),
        'urls': re.compile(r'http[s]?://\S+'),
        'emails': re.compile(r'[\w\.-]+@[\w\.-]+\.\w+'),
        'phones': re.compile(r'\(?\d{2}\)?\s*\d{4,5}-\d{4}'),
        'punctuation': re.compile(r'[^\w\s]')
    }

    def __init__(self):
        """Initialize NLP processor with Portuguese language resources."""
        self.stop_words = set(stopwords.words('portuguese'))
        self.stemmer = RSLPStemmer()
        self.min_token_length = 2

    def remove_accents(self, text: str) -> str:
        """Remove diacritical marks from text while preserving original characters."""
        return ''.join(
            c for c in unicodedata.normalize('NFD', text)
            if unicodedata.category(c) != 'Mn'
        )

    def clean_email_headers(self, text: str) -> str:
        """Remove standard email headers from the text."""
        for pattern in self.HEADER_PATTERNS:
            text = pattern.sub('', text)
        return text

    def _clean_text(self, text: str) -> str:
        """Apply all text cleaning operations."""
        text = self.remove_accents(text.lower())
        text = self.clean_email_headers(text)
        
        for pattern in self.CLEANING_PATTERNS.values():
            text = pattern.sub(' ', text)
            
        return text

    def tokenize_and_clean(self, text: str) -> List[str]:
        """
        Tokenize and clean text by:
        1. Normalizing and removing accents
        2. Removing headers and special patterns
        3. Stemming and filtering tokens
        """
        text = self._clean_text(text)
        tokens = nltk.word_tokenize(text)
        
        return [
            self.stemmer.stem(token)
            for token in tokens
            if (token not in self.stop_words and 
                len(token) > self.min_token_length)
        ]

    def extract_key_phrases(self, text: str, top_n: int = 5) -> List[str]:
        """
        Extract most frequent key phrases from text.
        
        Args:
            text: Input text to analyze
            top_n: Number of top phrases to return
            
        Returns:
            List of top key phrases ordered by frequency
        """
        tokens = self.tokenize_and_clean(text)
        freq_dist = nltk.FreqDist(tokens)
        return [word for word, _ in freq_dist.most_common(top_n)]
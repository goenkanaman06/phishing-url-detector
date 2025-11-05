import re
from urllib.parse import urlparse

def extract_features(url):
    """Extracts basic lexical features from a URL for phishing detection."""
    features = {}
    features['url_length'] = len(url)
    features['num_digits'] = sum(c.isdigit() for c in url)
    features['num_special'] = sum(1 for c in url if c in ['@','?','-','=','_'])
    features['https'] = 1 if url.startswith('https') else 0
    features['has_ip'] = 1 if re.search(r'(\d{1,3}\.){3}\d{1,3}', url) else 0
    features['contains_login'] = 1 if "login" in url else 0
    return features

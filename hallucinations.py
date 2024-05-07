# This can be the file where we outline the heuristics to catch content hallucinated by visual OCR models like Nougat. 

import re
from collections import Counter
from difflib import SequenceMatcher

def detect_word_repetition(text, threshold=5):
    """Detect excessive word repetition."""
    words = text.split()
    for i in range(len(words) - threshold):
        if len(set(words[i:i+threshold])) == 1:
            return True, ' '.join(words[i:i+threshold])
    return False, ""

def detect_phrase_repetition(text, phrase_length=4, threshold=7):
    """Detect excessive phrase repetition."""
    words = text.split()
    phrases = [' '.join(words[i:i+phrase_length]) for i in range(len(words) - phrase_length + 1)]
    phrase_counts = Counter(phrases)
    for phrase, count in phrase_counts.items():
        if count >= threshold:
            return True, phrase
    return False, ""

def calculate_repetition_ratio(text):
    """Calculate the ratio of unique words to total words."""
    words = text.split()
    unique_words = set(words)
    return len(unique_words) / len(words) if words else 0

def sequence_similarity(a, b):
    """Calculate similarity between two sequences."""
    return SequenceMatcher(None, a, b).ratio()

def detect_text_hallucination(text):
    word_repetition = detect_word_repetition(text)
    phrase_repetition = detect_phrase_repetition(text)
    repetition_ratio = calculate_repetition_ratio(text)
    # print(repetition_ratio)
    
    if len(text)<2:
        return False, "Too short for statistics"
    if word_repetition[0]:
        return True, f"Excessive word repetition detected: {word_repetition[1]}"
    elif phrase_repetition[0]:
        return True, f"Excessive phrase repetition detected: {phrase_repetition[1]}"
    elif repetition_ratio < 0.30:  # Arbitrary threshold for demonstration
        return True, "Low unique-to-total word ratio, indicating possible repetition."
    return False, "No significant repetition detected."
   
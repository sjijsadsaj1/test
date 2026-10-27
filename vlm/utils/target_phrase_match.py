"""Narrow normalization for DINO combining two synonymous picture nouns."""
import re

def target_phrase_matches(phrase, target_labels, mode='off'):
    # Default and other categories preserve the exact upstream membership test.
    if phrase in target_labels:
        return True
    if mode not in ('picture_phrase', 'picture_pose') or target_labels != ['picture', 'framed photograph']:
        return False
    words = re.findall(r'[a-z]+', phrase.lower())
    # Require BOTH nouns, reject all distractor words, never accept only
    # 'photograph' (often emitted for a box spanning the whole input image).
    return {'picture', 'photograph'} <= set(words) <= {'picture', 'photograph', 'framed'}

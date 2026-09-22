"""Word-frequency prediction, extending word_demo.py.

Loads the same english_words "web2" dictionary and wordfreq zipf
frequencies once at startup (so the cost of computing them is paid
one time, not on every keystroke), then answers two kinds of queries
by prefix as the user types:

  predict_next_letters(prefix) - which letter is most likely to come
      next, for the letter-highlighting on the keyboard.
  get_word_suggestions(prefix) - which whole words are most likely,
      for the word-suggestion sidebar.

Both use the exact approach from word_demo.py: filter the dictionary
to words starting with the prefix, weight each by zipf_frequency, and
rank by total weight.
"""

from english_words import get_english_words_set
from wordfreq import zipf_frequency

print("Loading dictionary...")
_words = get_english_words_set(["web2"], alpha=True, lower=True)
WORD_FREQ = {word: zipf_frequency(word, "en") for word in _words}
DICTIONARY = sorted(WORD_FREQ.keys())
print(f"Loaded {len(DICTIONARY)} words.")


def _matching_words(prefix):
    if not prefix:
        return DICTIONARY
    n = len(prefix)
    return [word for word in DICTIONARY if len(word) > n and word[:n] == prefix]


def predict_next_letters(prefix, limit=6):
    """Return [(letter, weight), ...] sorted by weight, most likely first."""
    prefix = prefix.lower()
    n = len(prefix)
    weights = {}
    for word in _matching_words(prefix):
        letter = word[n]
        weights[letter] = weights.get(letter, 0) + WORD_FREQ[word]
    return sorted(weights.items(), key=lambda pair: pair[1], reverse=True)[:limit]


def get_word_suggestions(prefix, limit=6):
    """Return the most common whole words starting with prefix."""
    prefix = prefix.lower()
    if not prefix:
        return []
    matches = _matching_words(prefix)
    return sorted(matches, key=lambda word: WORD_FREQ[word], reverse=True)[:limit]


if __name__ == "__main__":
    # Quick sanity check when run directly, e.g. `python predictor.py ha`
    import sys
    test_prefix = sys.argv[1] if len(sys.argv) > 1 else "ha"
    print(f"Next letters after {test_prefix!r}:", predict_next_letters(test_prefix))
    print(f"Word suggestions for {test_prefix!r}:", get_word_suggestions(test_prefix))

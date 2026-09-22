from english_words import get_english_words_set
from wordfreq import zipf_frequency

dictionary = list(get_english_words_set(["web2"], alpha=True, lower=True))

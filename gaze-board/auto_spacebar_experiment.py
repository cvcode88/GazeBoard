from english_words import get_english_words_set

dictionary = list(get_english_words_set(["web2"], alpha=True, lower=True))

auto_spacebar_enabled = True   # the optional toggle
delay = 0.3                    # the optional slider value, in seconds
current_letters = "example"    # their current letters in their word
words_count = 0

# MAKE SURE THIS FUNCTION BELOW IS RAN EVERY NEW LETTER THE USER TYPES

def auto_spacebar(current_word, dictionary, enabled, delay):
    if not enabled:
        return False

    current_word = current_word.lower()
    words_count = 0
    for i in range(len(dictionary)):
        if len(dictionary[i]) > len(current_word) and dictionary[i][:len(current_word)] == current_word:
            words_count += 1

    is_complete_word = current_word in dictionary
    return is_complete_word and words_count == 0


    


result = auto_spacebar("hello", dictionary, auto_spacebar_enabled, delay)
print(result)


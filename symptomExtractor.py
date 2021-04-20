import nltk
from nltk.tokenize import word_tokenize
sent = "Hello everyone. Recovering from 2nd Covid infection, this time around arthritis is so painful, unlike " \
       "anything I've experienced before. I searched posts in this amazing group and found so many people with " \
       "flaring or newly-diagnosed arthritis post-covid. Are there any studies or medical papers about Covid " \
       "causing or pain re-activating inflammatory arthritis? Gentle virtual hugs to all.."

text_tokens = word_tokenize(sent)
print(nltk.pos_tag(text_tokens))
for i in nltk.pos_tag(text_tokens):
    if i[1] == "NN":
        print(i[0])


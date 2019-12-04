# script to resegment and auto-correct documents in the No More Silence dataset.

''' Heavily lifted from Peter Norvig's:

Statistical Natural Language Processing in Python. 
or 
How To Do Things With Words. And Counters. 
or 
Everything I Needed to Know About NLP I learned From Sesame Street. 
Except Kneser-Ney Smoothing. 
The Count Didn't Cover That. 

availabel at: 
https://nbviewer.jupyter.org/url/norvig.com/ipython/How%20to%20Do%20Things%20with%20Words.ipynb

'''

'''
Logic for correction:

for all tokens, segment token, and spell correct all resulting new tokens.
spell correct will consider words that are two or less edits away (including space " ").

This will fail for words that are 3 edits away, long words with a space incorrectly inserted,
and chunks that need segemnted, but contain rare words and spelling errors.

Very bad for names and documents in spanish.
'''
import re
import math
import string
from collections import Counter
from __future__ import division

import spacy
import re
import pandas as pd

from time import time

# Instantiate spacy
nlp = spacy.load("en_core_web_lg")
df = pd.read_csv("NoMoreSilence_ProjectData.tsv", delimiter="\t")

with open('big.txt', encoding="utf8") as file: TEXT = file.read()
len(TEXT)

def tokens(text):
    "List all the word tokens (consecutive letters) in a text. Normalize to lowercase."
    return re.findall('[a-z]+', text.lower()) 

WORDS = tokens(TEXT)
stop = spacy.lang.en.stop_words.STOP_WORDS.union(set("abcdefghijklmnopqrstuvwxyz"))

COUNTS = Counter(WORDS)

def correct(word):
    "Find the best spelling correction for this word."
    # Prefer edit distance 0, then 1, then 2; otherwise default to word itself.
    candidates = (known(edits0(word)) or 
                  known(edits1(word)) or 
                  known(edits2(word)) or 
                  [word])
    #return max(candidates, key=COUNTS.get)
    #print(candidates)
    return max(candidates, key=ranker)

def ranker(candidate):
    #print(COUNTS2[candidate] if " " in candidate else COUNTS[candidate])
    #return COUNTS2[candidate] if " " in candidate else sum([COUNTS[c] for c in candidate.split() if c not in stop]) 
    #print(sum([COUNTS1[word] for word in candidate.split(" ")]) / len(candidate.split(" ")))
    #return sum([COUNTS1[word] for word in candidate.split(" ")]) / len(candidate.split(" "))
    return max((COUNTS2[candidate], sum([COUNTS[c] for c in candidate.split() if c not in stop]))) 

def known(words):
    "Return the subset of words that are actually in the dictionary."
    #return {w for w in words if w in COUNTS}
    return {w for w in words if all([ws in COUNTS for ws in w.split()])}
    #return {w for w in words if any([w in COUNTS, w in COUNTS2])}
    
def edits0(word): 
    "Return all strings that are zero edits away from word (i.e., just word itself)."
    return {word}

def edits2(word):
    "Return all strings that are two edits away from this word."
    return {e2 for e1 in edits1(word) for e2 in edits1(e1)}

def edits1(word):
    "Return all strings that are one edit away from this word."
    pairs      = splits(word)
    deletes    = [a+b[1:]           for (a, b) in pairs if b]
    transposes = [a+b[1]+b[0]+b[2:] for (a, b) in pairs if len(b) > 1]
    replaces   = [a+c+b[1:]         for (a, b) in pairs for c in alphabet if b]
    inserts    = [a+c+b             for (a, b) in pairs for c in alphabet]
    return set(deletes + transposes + replaces + inserts)

def splits(word):
    "Return a list of all possible (first, rest) pairs that comprise word."
    return [(word[:i], word[i:]) 
            for i in range(len(word)+1)]

alphabet = 'abcdefghijklmnopqrstuvwxyz '

def correct_text(text):
    "Correct all the words within a text, returning the corrected text."
    return re.sub('[a-zA-Z]+', correct_match, text)

def correct_match(match):
    "Spell-correct word in match, and preserve proper upper/lower/title case."
    word = match.group()
    return case_of(word)(correct(word.lower()))

def case_of(text):
    "Return the case-function appropriate for text: upper, lower, title, or just str."
    return (str.upper if text.isupper() else
            str.lower if text.islower() else
            str.title if text.istitle() else
            str)

def load_counts(filename, sep='\t'):
    """Return a Counter initialized from key-value pairs, 
    one on each line of filename."""
    C = Counter()
    for line in open(filename):
        key, count = line.split(sep)
        C[key] = int(count)
    return C

COUNTS2 = load_counts('count_2w.txt')
COUNTS1 = load_counts('count_1w.txt')

def pdist_good_turing_hack(counter, onecounter, base=1/26., prior=1e-8):
    """The probability of word, given evidence from the counter.
    For unknown words, look at the one-counts from onecounter, based on length.
    This gets ideas from Good-Turing, but doesn't implement all of it.
    prior is an additional factor to make unknowns less likely.
    base is how much we attenuate probability for each letter beyond longest."""
    N = sum(list(counter.values()))
    N2 = sum(list(onecounter.values()))
    lengths = map(len, [w for w in onecounter if onecounter[w] == 1])
    ones = Counter(lengths)
    longest = max(ones)
    return (lambda word: 
            counter[word] / N if (word in counter) 
            else prior * (ones[len(word)] / N2 or 
                          ones[longest] / N2 * base ** (len(word)-longest)))

# Redefine P1w
P1w = pdist_good_turing_hack(COUNTS1, COUNTS)


def pdist(counter):
    "Make a probability distribution, given evidence from a Counter."
    N = sum(list(counter.values()))
    return lambda x: counter[x]/N

Pword = pdist(COUNTS)

def Pwords(words):
    "Probability of words, assuming each word is independent of others."
    #return product(Pword(w) for w in words)
    return product(Pword(w) for w in words)

def product(nums):
    "Multiply the numbers together.  (Like `sum`, but with multiplication.)"
    result = 1
    for x in nums:
        result *= x
    return result

def memo(f):
    "Memoize function f, whose args must all be hashable."
    cache = {}
    def fmemo(*args):
        if args not in cache:
            cache[args] = f(*args)
        return cache[args]
    fmemo.cache = cache
    return fmemo

def splits(text, start=0, L=20):
    "Return a list of all (first, rest) pairs; start <= len(first) <= L."
    return [(text[:i], text[i:]) 
            for i in range(start, min(len(text), L)+1)]

@memo
def segment(text):
    "Return a list of words that is the most probable segmentation of text."
    if not text: 
        return []
    else:
        candidates = ([first] + segment(rest) 
                      for (first, rest) in splits(text, 1))
        return max(candidates, key=Pwords)

# Change Pwords to use P1w (the bigger dictionary) instead of Pword
def Pwords(words):
    "Probability of words, assuming each word is independent of others."
    return product(P1w(w) for w in words)

# regex to remove page numbers
regex = re.compile('pgNbr=\d+')

# Remove weird chars
regex2 = re.compile('¬ ') 
regex3 = re.compile("[^0-9a-zA-Z:,\. ;!#%?]+")

# list of sentences
df["Corrected Text"] = ""
for i,text in enumerate(df["Ocr text"]):
    try:
        t0 = time()

        print("Processing document: " + str(i))
        print("\tCharacters: " + str(len(text)))

        # Remove page numbers
        text = re.sub(regex, "", text)
        text = re.sub(regex2, "", text)
        text = re.sub(regex3, '', text)
        
        # add space after punctuation 
        text = re.sub(r'(?<=[.,!?])(?=[^\s])', r' ', text)

        # segment sentences
        doc = nlp(text)

        print("\tSentences: " + str(len(list(doc.sents))))

        # for each sentence get tokesn and remove stop words
        j = 0
        fixed = ""
        for sent in doc.sents:
            j += 1
            print(sent)
            #print("\tProcesing sentence:" + str(j), end = "\r")
            words = [segment(word) for word in sent.text.lower().split()]
            words = [correct_text(word) for sublist in words for word in sublist]
            sent = " ".join(words)
            # remove space before punctuation
            sent = re.sub(r'\s([?.!,](?:\s|$))', r'\1', sent)
            sent = re.sub(r'\s([?.!,](?:\s|$))', r'\1', sent)
            # remove space between numbers
            sent = re.sub('(?<=[\d,]) (?=[\d,])', '', sent)
            print(sent)
            print()
            fixed += " " + sent
        segment.cache.clear()
        print()
        print("\tProcess Time:" + str(time() - t0))

        df["Corrected Text"][i] += fixed
        df.to_csv("NoMoreSilence_ProjectDataV2.tsv", sep="\t")
    except:
        pass



import requests
import contractions
from nltk.tokenize import sent_tokenize, word_tokenize
import math
import re


class LanguageModel:
    def __init__(self, n_gram=2, ):
        self.n = n_gram

    def bigrams(self, word_list):
        bigrams = []
        for j in range(len(word_list) - self.n + 1):
            bigrams.append(tuple(word_list[j:j + self.n]))
        return bigrams

    def add_stop_symbol(self, word_list):
        for index, word in enumerate(word_list):
            if re.match(r"[.!?]+(\")*", word):
                word_list[index] = '<s>'
                word_list.insert(index, '</s>')
        word_list.insert(0, word_list.pop())
        return word_list


    def build_model(self, train_text):
        self.number_of_sentences = len(sent_tokenize(train_text))
        word_list = word_tokenize(train_text)
        token_count = {}
        for index, word in enumerate(word_list):
            if word not in token_count:
                token_count[word] = 1
            else:
                token_count[word] += 1
        self.token_dict = token_count
        self.total_count =  sum(self.token_dict.values()) + self.number_of_sentences
        self.token_dict['<UNK>'] = 0
        for key in self.token_dict:
            if self.token_dict[key] < 1:
                self.token_dict['<UNK>'] += 1
        word_list = self.add_stop_symbol(word_list)
        self.bigram_train_dict = {}
        for bigram in self.bigrams(word_list):
            if bigram not in self.bigram_train_dict:
                self.bigram_train_dict[bigram] = 1
            else:
                self.bigram_train_dict[bigram] += 1

    def bigram_prob_sentence(self, sentence):
        sentence_prob = 0
        for bigram in self.bigrams(sentence):

            if bigram not in self.bigram_train_dict:
                test_word = bigram[1]
                if test_word not in self.token_dict:
                    test_word = '<UNK>'

                numerator = self.token_dict[test_word]
                denominator = self.total_count

            elif '<s>' == bigram[0] or '</s>' == bigram[0]:
                numerator = self.bigram_train_dict[bigram]
                denominator = self.number_of_sentences

            else:
                numerator = self.bigram_train_dict[bigram]
                denominator = self.token_dict[bigram[0]]

            bigram_prob = numerator / float(denominator)
            sentence_prob += math.log(bigram_prob)

        return math.exp(sentence_prob)

    def calculate_proba(self, test_sentence):
        print("Sentence ", ': ', test_sentence)
        sentence = word_tokenize(test_sentence)
        sentence = self.add_stop_symbol(sentence)
        bigram_prob = self.bigram_prob_sentence(sentence)
        print('Bigram probability', bigram_prob)


test_text = "Our business is not unknown to the boss."
responce = requests.get("https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt")
train_text = contractions.fix(responce.text)
model = LanguageModel()
model.build_model(train_text)
model.calculate_proba(test_text)

import re
import sys
import random
import math
import collections
import nltk
from collections import Counter


class Spell_Checker:
    """The class implements a context sensitive spell checker. The corrections
        are done in the Noisy Channel framework, based on a language model and
        an error distribution model.
    """

    def __init__(self, lm=None):
        """Initializing a spell checker object with a language model as an
        instance  variable.

        Args:
            lm: a language model object. Defaults to None.
        """
        self.lm = lm

    def add_language_model(self, lm):
        """Adds the specified language model as an instance variable.
            (Replaces an older LM dictionary if set)

            Args:
                lm: a Spell_Checker.Language_Model object
        """
        self.lm = lm

    def add_error_tables(self, error_tables):
        """ Adds the specified dictionary of error tables as an instance variable.
            (Replaces an older value dictionary if set)

            Args:
            error_tables (dict): a dictionary of error tables in the format
            of the provided confusion matrices:
            https://www.dropbox.com/s/ic40soda29emt4a/spelling_confusion_matrices.py?dl=0
        """
        self.error_tables = error_tables

    def evaluate_text(self, text):
        """Returns the log-likelihood of the specified text given the language
            model in use. Smoothing should be applied on texts containing OOV words

           Args:
               text (str): Text to evaluate.

           Returns:
               Float. The float should reflect the (log) probability.
        """
        return self.lm.evaluate_text(text)

    def get_candidates(self, x):
        all_chars = 'absdefghijklmnopqrstuvwxyz'

        candidates = set()

        # check deletion
        for i in range(len(x) + 1):
            for c in all_chars:
                w = x[:i] + c + x[i:]
                type_error = 'deletion'
                key = w[i - 1] + w[i] if i > 0 else '#' + w[i]
                candidates.add(((type_error, key), w))

        # check Insertion
        for i in range(len(x)):
            w = x[:i] + x[i + 1:]
            if w == "": continue
            type_error = 'insertion'
            key = w[i - 1] + x[i] if i > 0 else '#' + x[i]
            candidates.add(((type_error, key), w))

        # check Substitution
        for i in range(len(x)):
            for c in all_chars:
                if x[i] == c: continue
                w = x[:i] + c + x[i + 1:]
                type_error = 'substitution'
                key = x[i] + w[i]
                candidates.add(((type_error, key), w))

        # check Transposition
        for i in range(len(x) - 1):
            if x[i] == x[i + 1]: continue
            w = x[:i] + x[i + 1] + x[i] + x[i + 2:]
            type_error = 'transposition'
            key = w[i] + w[i + 1]
            candidates.add(((type_error, key), w))

        return candidates

    def p_typo(self, error_type, key):
        up = self.error_tables[error_type][key]
        down = 0
        key_for_char = key.replace("#", " ")
        if error_type == 'deletion':
            down = self.lm.pairs_dict.get(key_for_char, 0)
        elif error_type == 'insertion':
            down = self.lm.chars_dict.get(key_for_char[0], 0)
        elif error_type == 'substitution':
            down = self.lm.chars_dict.get(key_for_char[1], 0)
        elif error_type == 'transposition':
            down = self.lm.pairs_dict.get(key_for_char, 0)

        if down == 0:
            down += 1
        p = up / down
        return p

    def get_candidate_with_p(self, x):
        candidates_with_oov = self.get_candidates(x)
        c_with_p = dict()
        c_with_p_with_oov = dict()
        c2_with_p = dict()

        # 1 distance
        for (error_type, key), c in candidates_with_oov:
            p = self.p_typo(error_type, key)
            if (c,) in self.lm.model_dict_1gram:
                if c not in c_with_p: c_with_p[c] = 0
                c_with_p[c] = max(c_with_p[c], p)
            if c not in c_with_p_with_oov: c_with_p_with_oov[c] = 0

            c_with_p_with_oov[c] = max(c_with_p_with_oov[c], p)

        # 2 distance
        for c, p1 in c_with_p_with_oov.items():
            candidates2 = self.get_candidates(c)
            for (error_type, key), c2 in candidates2:
                p2 = self.p_typo(error_type, key)
                if (c2,) in self.lm.model_dict_1gram:
                    p = p1 * p2
                    if c2 not in c2_with_p: c2_with_p[c2] = 0
                    c2_with_p[c2] = max(c2_with_p[c2], p)

        keys = set(c_with_p.keys()).union(c2_with_p.keys())
        c_with_p_merge = {k: max(c_with_p.get(k, float('-inf')), c2_with_p.get(k, float('-inf'))) for k in keys}
        return c_with_p_merge

    def tokens_to_text(self, tokens):
        text = ""
        for sent in tokens:
            text_sent = " ".join(sent) + ". "
            text += text_sent
        text = text.strip()
        return text

    def spell_check(self, text, alpha=0.95):
        """ Returns the most probable fix for the specified text. Use a simple
            noisy channel model if the number of tokens in the specified text is
            smaller than the length (n) of the language model.

            Args:
                text (str): the text to spell check.
                alpha (float): the probability of keeping a lexical word as is.

            Return:
                A modified string (or a copy of the original if no corrections are made.)
        """
        text_norm = normalize_text(text)
        tokens = self.lm.get_tokens(text_norm, self.lm.chars)

        for i in range(len(tokens)):
            change_sent_indx = change_word_indx = new_word = None
            changed_be_made = False

            sent = tokens[i]
            if len(sent) == 0: continue

            p_max_sent = math.log(alpha) + self.lm.evaluate_text(self.tokens_to_text([sent]))

            for j in range(len(sent)):
                word = sent[j]
                candidates = self.get_candidate_with_p(word)
                c_dict = {}
                for c, p_error in candidates.items():
                    if c == word or p_error <= 0: continue
                    new_tokens = [tmp.copy() for tmp in tokens]
                    new_tokens[i][j] = c
                    tokens_text = self.tokens_to_text([new_tokens[i]])
                    p_eval = self.lm.evaluate_text(tokens_text)
                    if c not in c_dict:
                        c_dict[c] = (-math.inf, -math.inf)
                    c_dict[c] = (max(p_error, c_dict[c][0]), p_eval)
                sum_p_error = sum([i[0] for i in c_dict.values()])
                for c, (p_error, p_eval) in c_dict.items():
                    total_p = math.log(1 - alpha) + math.log(p_error / sum_p_error) + p_eval
                    if total_p >= p_max_sent:
                        change_sent_indx = i
                        change_word_indx = j
                        new_word = c
                        changed_be_made = True
                        p_max_sent = total_p
            if changed_be_made:
                tokens[change_sent_indx][change_word_indx] = new_word
        result = self.tokens_to_text(tokens)
        return result

    #####################################################################
    #                   Inner class                                     #
    #####################################################################

    class Language_Model:
        """The class implements a Markov Language Model that learns a model from a given text.
            It supports language generation and the evaluation of a given string.
            The class can be applied on both word level and character level.
        """

        def __init__(self, n=3, chars=False):
            """Initializing a language model object.
            Args:
                n (int): the length of the markov unit (the n of the n-gram). Defaults to 3.
                chars (bool): True iff the model consists of ngrams of characters rather than word tokens.
                              Defaults to False
            """
            self.n = n
            self.model_dict = None  # a dictionary of the form {ngram:count}, holding counts of all ngrams in the specified text.
            # NOTE: This dictionary format is inefficient and insufficient (why?), therefore  you can (even encouraged to)
            # use a better data structure.
            # However, you are requested to support this format for two reasons:
            # (1) It is very straight forward and force you to understand the logic behind LM, and
            # (2) It serves as the normal form for the LM so we can call get_model_dictionary() and peek into you model.
            self.chars = chars
            self.padding = True

        def get_tokens(self, text, chars=False):

            sentences = text.split(".")  # split to Sentences

            sentences = [s.strip() for s in sentences]

            if chars:
                tokens = [list(sentence) for sentence in sentences]  # split to Characters
            else:
                tokens = [sentence.split(" ") for sentence in sentences]  # split to Words

            if tokens[-1] == ['']:
                tokens = [sent for sent in tokens if sent != ['']]
            return tokens

        def add_padding(self, tokens, before, after):
            tokens_with_pad = [['<s>'] * before + sent + ['</s>'] * min(after, self.n - 1) for sent in tokens]
            return tokens_with_pad

        def build_model(self, text):  # should be called build_model
            """populates the instance variable model_dict.

                Args:
                    text (str): the text to construct the model from.
            """

            text_norm = normalize_text(text)
            tokens = self.get_tokens(text_norm, self.chars)

            self.pairs_dict = Counter()
            self.chars_dict = Counter()
            self.model_dict = Counter()
            self.model_dict_minus_1 = dict()
            self.model_dict_1gram = Counter()

            # populate ngram
            for sent in tokens:
                cur = sent

                if self.padding:
                    cur = ['<s>'] * (self.n - 1) + sent + ['</s>']

                for i in range(len(cur) - self.n + 1):
                    ngram = tuple(cur[i:i + self.n])
                    self.model_dict[ngram] += 1

            # populate ngram_minus_1
            for sent in tokens:
                cur = sent

                if self.padding:
                    cur = ['<s>'] * (self.n - 1) + sent + ['</s>']

                for i in range(len(cur) - self.n + 1):
                    ngram = tuple(cur[i:i + self.n])
                    if ngram[:-1] not in self.model_dict_minus_1:
                        self.model_dict_minus_1[ngram[:-1]] = Counter()
                    self.model_dict_minus_1[ngram[:-1]][ngram[-1]] += 1

            # populate 1gram
            for sent in tokens:
                for word in sent:
                    self.model_dict_1gram[tuple([word])] += 1

            # populate pairs_dict
            for sentence in tokens:
                for word in sentence:
                    if len(word) == 0: continue
                    for i in range(len(word) - 1):
                        self.pairs_dict[word[i] + word[i + 1]] += 1
                    self.pairs_dict[" " + word[0]] += 1
                    self.pairs_dict[word[-1] + " "] += 1

            # populate chars_dict
            for sentence in tokens:
                for word in sentence:
                    if len(word) == 0: continue
                    self.chars_dict[" "] += 1
                    for i in range(len(word)):
                        self.chars_dict[word[i]] += 1

            self.pairs_dict = dict(self.pairs_dict)
            self.chars_dict = dict(self.chars_dict)
            self.model_dict = dict(self.model_dict)
            self.model_dict_minus_1 = {k: dict(v) for k, v in self.model_dict_minus_1.items()}
            self.model_dict_1gram = dict(self.model_dict_1gram)
            self.V = len(self.model_dict)

        def get_model_dictionary(self):
            """Returns the dictionary class object
            """
            return self.model_dict

        def get_model_window_size(self):
            """Returning the size of the context window (the n in "n-gram")
            """
            return self.n

        def generate(self, context=None, n=20):
            """Returns a string of the specified length, generated by applying the language model
            to the specified seed context. If no context is specified the context should be sampled
            from the models' contexts distribution. Generation should stop before the n'th word if the
            contexts are exhausted. If the length of the specified context exceeds (or equal to)
            the specified n, the method should return a prefix of length n of the specified context.

                Args:
                    context (str): a seed context to start the generated string from. Defaults to None
                    n (int): the length of the string to be generated.

                Return:
                    String. The generated text.

            """
            if context is None or context == '':
                tokens = [[]]
            else:
                text_norm = normalize_text(context)
                tokens = self.get_tokens(text_norm, self.chars)

            if sum([len(sent) for sent in tokens]) >= n:
                result = []
                total_words = 0
                for sent in tokens:
                    res_sent = []
                    for word in sent:
                        if total_words >= 20:
                            result.append(res_sent)
                            return self.tokens_to_text(result)
                        res_sent.append(word)
                        total_words += 1
                    result.append(res_sent)

            while sum([len(sent) for sent in tokens]) < n:
                last_ngram = tokens[-1][-self.n:]
                last_ngram = ['<s>'] * (self.n - len(last_ngram)) + last_ngram
                last_ngram = tuple(last_ngram[-self.n + 1:])
                if last_ngram not in self.model_dict_minus_1 and self.n != 1:
                    break
                cur_dict = self.model_dict_minus_1[last_ngram] if self.n != 1 else self.model_dict
                if self.n == 1:
                    cur_dict = {k[0]: v for k, v in cur_dict.items()}
                word = random.choices(list(cur_dict.keys()), list(cur_dict.values()))
                while word == ('',) or word == '' or word == '\s':
                    word = random.choices(list(cur_dict.keys()), list(cur_dict.values()))
                if word[0] != '</s>':
                    tokens[-1].append(word[0])
                else:
                    tokens.append([])
            result = self.tokens_to_text(tokens)
            return result

        def tokens_to_text(self, tokens):
            text = ""
            for sent in tokens:
                text_sent = " ".join(sent) + ". "
                text += text_sent
            text = text.strip()
            return text

        def evaluate_text(self, text):
            """Returns the log-likelihood of the specified text to be a product of the model.
               Laplace smoothing should be applied if necessary.

               Args:
                   text (str): Text to evaluate.

               Returns:
                   Float. The float should reflect the (log) probability.
            """
            text_norm = normalize_text(text)
            tokens = self.get_tokens(text_norm, self.chars)

            if self.padding:
                tokens = self.add_padding(tokens, self.n - 1, 1)

            total_prob = 0
            for sent in tokens:
                for i in range(len(sent) - self.n + 1):
                    ngram = tuple(sent[i:i + self.n])
                    if ngram in self.model_dict:
                        up = self.model_dict[ngram]
                        down = sum(self.model_dict_minus_1[ngram[:-1]].values())
                        prob = up / down
                    else:
                        prob = self.smooth(ngram)
                    total_prob += math.log(prob)
            return total_prob

        def smooth(self, ngram):
            """Returns the smoothed (Laplace) probability of the specified ngram.

                Args:
                    ngram (str): the ngram to have its probability smoothed

                Returns:
                    float. The smoothed probability.
            """
            up = max(self.model_dict.get((ngram,), 0), self.model_dict.get(ngram, 0))
            down = sum(self.model_dict_minus_1[ngram[:-1]].values()) if ngram[
                                                                        :-1] in self.model_dict_minus_1 else 0
            return (up + 1) / (down + self.V)


def normalize_text(text):
    """Returns a normalized version of the specified string.
      You can add default parameters as you like (they should have default values!)
      You should explain your decisions in the header of the function.

      Args:
        text (str): the text to normalize

      Returns:
        string. the normalized text.
    """

    if text == None or len(text) == 0:
        text = ""

    clean_text = text.lower()
    clean_text = clean_text.replace('\n', ' ')
    clean_text = re.sub(r'[^\w\s\n.]', '', clean_text)
    clean_text = re.sub(r'\s+', ' ', clean_text)

    return clean_text


def who_am_i():  # this is not a class method
    """Returns a ductionary with your name, id number and email. keys=['name', 'id','email']
        Make sure you return your own info!
    """
    return {'name': 'Oryan Yehezkel', 'id': '311495824', 'email': 'oryanyeh@post.bgu.ac.il'}

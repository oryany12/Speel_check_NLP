from spelling_confusion_matrices import error_tables
from Spell_check import *

# """
# ===== Load Corpus
# """
# f = open("Corpus/big.txt", "r")
# corpus = f.read()
# corpus = normalize_text(corpus)
# mini_corpus = corpus
# # mini_corpus = 'the project is good'
#
#
# """
# Create Language_Model Obj + Tests
# """
# lm = Spell_Checker.Language_Model(n=3, chars=False)
# lm.build_model(mini_corpus)
# print(lm.evaluate_text("the project is"))
# a=0
# # print(lm.generate('the good', n=10))
#
#
# """
# Create Spell_Checker Obj + Tests
# """
#
# checker = Spell_Checker()
# checker.add_language_model(lm)
# checker.add_error_tables(error_tables)
# print(checker.spell_check("evident", 0.95))
# print(checker.spell_check("evidently anxoius to disperse", 0.95))
# print(checker.spell_check("evidently anxous to disperse", 0.95))
# print(checker.spell_check("evidently anxoius to disperse. the project is beautifu", 0.95))


LM = Spell_Checker()
while True:
    for i in range(10):
        print("===== N = ", i + 1, "=====")
        LM.add_language_model(Spell_Checker.Language_Model(n=i + 1))
        corpus = open("Corpus/big.txt", "r").read()
        # corpus = open("Corpus/corpus.data", "r").read()
        LM.lm.build_model(corpus)
        LM.add_error_tables(error_tables)

        text = 'we have seldom heard him mention her under any other nama. in his eyes she eclipses ana predominates the whole of her sex'
        text2 = 'we have seldom heard him mention her under any others'
        # text = 'we have seldom heard him mention her under any other nama.'
        # text = "evidently anxoius to disperse. the project is beautifu"
        # text = "nama"
        print(LM.spell_check(text))
        print(LM.evaluate_text(text))
        print(LM.lm.generate(n=20))
        print(LM.lm.generate(context=text2, n=20))

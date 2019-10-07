import os
import math
import json
import re
import csv

os.getcwd()

path_pos = "data/train/pos"
path_neg = "data/train/neg"

listing_pos = os.listdir(path_pos)
listing_neg = os.listdir(path_neg)

#print (listing)

pos_words = {}
neg_words = {}
c_pos = 0
c_neg = 0
for file_pos, file_neg in zip(listing_pos, listing_neg):

    f_pos = open(path_pos+"/"+file_pos, "r", encoding = "utf-8")
    f_neg = open(path_neg+"/"+file_neg, "r", encoding = "utf-8")

    allwords_pos = f_pos.read().split()
    allwords_neg = f_neg.read().split()
    

    for word_pos, word_neg in zip(allwords_pos, allwords_neg):
        #data cleaning
        word_pos_clean = re.sub(r"[-()\"#/@;:<>{}`+=~|.!?,]", "", word_pos)
        word_pos_clean = re.sub(r"[^\w\s]","",word_pos_clean)
        word_pos_clean = word_pos_clean.lower()

        word_neg_clean = re.sub(r"[-()\"#/@;:<>{}`+=~|.!?,]", "", word_neg)
        word_neg_clean = re.sub(r"[^\w\s]","",word_neg_clean)
        word_neg_clean = word_neg_clean.lower()

        count_pos = 0
        count_neg = 0

        if word_pos_clean.isalnum():
            value_pos = pos_words.get(word_pos)

            # calculate the frequency of positive words
            if value_pos:
                count_pos = value_pos
                count_pos = count_pos + 1
            else:
                count_pos = 1
            word_pos = word_pos_clean
            pos_words[word_pos] = count_pos

            #calculate total words, including duplicates
            c_pos = c_pos + count_pos

        if word_neg_clean.isalnum():
            value_neg = neg_words.get(word_neg)

            # calculate the frequency of negative words
            if value_neg:
                count_neg = value_neg
                count_neg = count_neg + 1
            else:
                count_neg = 1
            word_neg = word_neg_clean
            neg_words[word_neg] = count_neg

            #calculate total words, including duplicates
            c_neg = c_neg + count_neg

    f_pos.close()
    f_neg.close()

prob_w_pos = {}
prob_w_neg = {}

#total number of words in the vocabulary for positive and negative reviews
V_pos = len(pos_words)
V_neg = len(neg_words)

den_pos = c_pos + V_pos
den_neg = c_neg + V_neg

p_pos_total = 0
p_neg_total = 0

for key_pos, key_neg in zip(pos_words, neg_words):
    value_pos = pos_words[key_pos]
    value_neg = neg_words[key_neg]

    prob_pos = ((value_pos)+1)/den_pos
    #calculate total probability for positive words
    p_pos_total = p_pos_total + prob_pos

    prob_neg = ((value_neg)+1)/den_neg
    #calculate total probability for negative words
    p_neg_total = p_neg_total + prob_neg   

    prob_w_pos[key_pos] = prob_pos
    prob_w_neg[key_neg] = prob_neg


#now the model is ready, use the test data to get the results

path_test_pos = "data/test/pos"
path_test_neg = "data/test/neg"

listing_test_pos = os.listdir(path_test_pos)
listing_test_neg = os.listdir(path_test_neg)

pos_count_prob = 0
neg_count_prob = 0

total_pos_review = len(listing_test_pos)
total_neg_review = len(listing_test_neg)


for file_test_pos, file_test_neg in zip(listing_test_pos, listing_test_neg):
    f_test_pos = open(path_test_pos+"/"+file_test_pos, "r", encoding = "utf-8")

    f_test_neg = open(path_test_neg+"/"+file_test_neg, "r", encoding = "utf-8")

    allwords_test_pos = f_test_pos.read().split()
    allwords_test_neg = f_test_neg.read().split()

    calc_prob_pos = math.log(p_pos_total)
    calc_prob_neg = math.log(p_neg_total)

    for word_test_pos in allwords_test_pos:

        #data cleaning
        word_pos_clean = re.sub(r"[-()\"#/@;:<>{}`+=~|.!?,]", "", word_test_pos)
        word_pos_clean = re.sub(r"[^\w\s]","",word_pos_clean)
        word_test_pos = word_pos_clean
        word_test_pos = word_test_pos.lower()

        #storing the value of keys which will be used further
        key_pos = prob_w_pos.get(word_test_pos)
        key_neg = prob_w_neg.get(word_test_pos)

        if word_test_pos.isalnum():
            if key_pos:
                value_pos = prob_w_pos[word_test_pos]
                log_pos = math.log(value_pos)
                calc_prob_pos = calc_prob_pos + log_pos

            if key_neg:
                value_neg = prob_w_neg[word_test_pos]
                log_neg = math.log(value_neg)
                calc_prob_neg = calc_prob_neg + log_neg

        
    if calc_prob_pos >= calc_prob_neg:
        pos_count_prob = pos_count_prob + 1

    #re-initializing the value of P(c) for both positive and negative reviews
    calc_prob_pos = math.log(p_pos_total)
    calc_prob_neg = math.log(p_neg_total)

    #calculate the right placement for negative reviews test data
    for word_test_neg in allwords_test_neg:
        #data cleaning
        word_neg_clean = re.sub(r"[-()\"#/@;:<>{}`+=~|.!?,]", "", word_test_neg)
        word_neg_clean = re.sub(r"[^\w\s]","",word_neg_clean)
        word_test_neg = word_neg_clean
        word_test_neg = word_test_neg.lower()

        #storing the value of keys which will be used further
        key_pos = prob_w_pos.get(word_test_neg)
        key_neg = prob_w_neg.get(word_test_neg)

        if word_test_neg.isalnum():          
            if key_neg:
                value_neg = prob_w_neg[word_test_neg]
                log_neg = math.log(value_neg)
                calc_prob_neg = calc_prob_neg + log_neg

            if key_pos:
                value_pos = prob_w_pos[word_test_neg]
                log_pos = math.log(value_pos)
                calc_prob_pos = calc_prob_pos + log_pos


    if calc_prob_neg >= calc_prob_pos:
        neg_count_prob = neg_count_prob + 1

print("percentage of documents correctly classified for positive reviews:")
print((pos_count_prob/total_pos_review)*100)

print("percentage of documents correctly classified for negative reviews:")
print((neg_count_prob/total_neg_review)*100)
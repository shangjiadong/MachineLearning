import os
import sys
from collections import defaultdict
from __future__ import division
from math import log
startsym, stopsym = "<s>", "</s>"
import random
import time
import matplotlib.pyplot as plt
os.chdir("C:/MachineLearning/pj3/pj3-data")


def readfile(filename):
    for line in open(filename):
        wordtags = map(lambda x: x.rsplit("/", 1), line.split())
        yield [w for w,t in wordtags], [t for w,t in wordtags] # (word_seq, tag_seq) pair
    
def mle(filename): # Max Likelihood Estimation of HMM
    twfreq = defaultdict(lambda : defaultdict(int))
    ttfreq = defaultdict(lambda : defaultdict(int)) 
    tagfreq = defaultdict(int)    
    dictionary = defaultdict(set)

    for words, tags in readfile(filename):
        last = startsym
        tagfreq[last] += 1
        for word, tag in zip(words, tags) + [(stopsym, stopsym)]:
            #if tag == "VBP": tag = "VB" # +1 smoothing
            twfreq[tag][word] += 1            
            ttfreq[last][tag] += 1
            dictionary[word].add(tag)
            tagfreq[tag] += 1
            last = tag            
    
    model = defaultdict(float)
    num_tags = len(tagfreq)
    for tag, freq in tagfreq.iteritems(): 
        logfreq = log(freq)
        for word, f in twfreq[tag].iteritems():
            model[tag, word] = log(f) - logfreq 
        logfreq2 = log(freq + num_tags)
        for t in tagfreq: # all tags
            model[tag, t] = log(ttfreq[tag][t] + 1) - logfreq2 # +1 smoothing
        
    return dictionary, model

def decode(words, dictionary, model):

    def backtrack(i, tag):
        if i == 0:
            return []
        return backtrack(i-1, back[i][tag]) + [tag]

    words = [startsym] + words + [stopsym]

    best = defaultdict(lambda: defaultdict(lambda: float("-inf")))
    best[0][startsym] = 1
    back = defaultdict(dict)

    #print " ".join("%s/%s" % wordtag for wordtag in zip(words,tags)[1:-1])
    for i, word in enumerate(words[1:], 1):
        for tag in dictionary[word]:
            for prev in best[i-1]:
                score = best[i-1][prev] + model[prev, tag] + model[tag, word] 
                if score > best[i][tag]:
                    best[i][tag] = score
                    back[i][tag] = prev
        #print i, word, dictionary[word], best[i]
    #print best[len(words)-1][stopsym]
    mytags = backtrack(len(words)-1, stopsym)[:-1]
    #print " ".join("%s/%s" % wordtag for wordtag in mywordtags)
    return mytags

def test(filename, dictionary, model):    
    
    errors = tot = 0
    for words, tags in readfile(filename):
        mytags = decode(words, dictionary ,model)
        errors += sum(t1!=t2 for (t1,t2) in zip(tags, mytags))
        tot += len(words) 
        
    return errors/tot        

trainfile = 'train.txt.lower.unk'
devfile = 'dev.txt.lower.unk'

dictionary, model = mle(trainfile)
print "train_err {0:.2%}".format(test(trainfile, dictionary, model))
print "dev_err {0:.2%}".format(test(devfile, dictionary, model))

### Perceptron

def perceptron(trainfile, devfile, dictionary, epochs = 5):           
    
    model = defaultdict(float)
    features = defaultdict(int)
    train_set = list(readfile(trainfile))
    whole = sum(map(lambda (x, y): len(x), train_set))
    train_percep, dev_percep = [], []
    for epoch in range(1, epochs + 1):
        #random.shuffle(train_set)
        errs = errwords = 0
        for words, tags in readfile(trainfile):
            mytag = decode(words, dictionary, model)
            if mytag != tags:
                errs += 1
                words = [startsym] + words + [stopsym]
                tags = [startsym] + tags + [stopsym]
                mytags = [startsym] + mytag + [stopsym]
                for i, (x, y, z) in enumerate(zip(words, tags, mytags)[1:], 1):
                    if y != z:
                        errwords += 1
                        model[y, x] += 1
                        model[z, x] -= 1
                        features[y, x] = 1
                        features[z, x] = 1
                        
                    if (y != z) or (tags[i-1] != mytags[i-1]):
                        model[tags[i-1], y] += 1
                        model[mytags[i-1], z] -= 1
                        features[tags[i-1], y] = 1
                        features[mytags[i-1], z] = 1
                        features[tags[i-1], y] = 1
                        features[mytags[i-1], z] = 1
                        
        dev_err = test(devfile, dictionary, model)
        num_feature = sum(features[feature] for feature in features)
        print "epoch %d, updates %d, |w| = %d, train_err %2.f%%, dev_err %.2f%%" %(epoch, errs, num_feature, float(errwords/whole)*100, dev_err*100)
        train_percep.append(float(errwords/whole)*100)
        dev_percep.append(dev_err * 100)
    return train_percep, dev_percep

trainfile = 'train.txt.lower.unk'
devfile = 'dev.txt.lower.unk'
dictionary, _ = mle(trainfile)
#print dictionary
t = time.time()
train_err, dev_err = perceptron(trainfile, devfile, dictionary, epochs = 5)
t1 = time.time()
print t1-t

### Average perceptron

def avg_perceptron(trainfile, devfile, dictionary, epochs = 10):           
    model = defaultdict(float)
    model_0 = defaultdict(float)
    model_c = defaultdict(float)
    features = defaultdict(int)
    c = 1
    train_set = list(readfile(trainfile))
    whole = sum(map(lambda (x, y): len(x), train_set))
    train_avg_percep, dev_avg_percep = [], []
    for epoch in range(1, epochs + 1):
        #random.shuffle(train_set)
        errs = errwords = 0
        for words, tags in train_set:
            mytags = decode(words, dictionary, model_0)
            if mytags != tags:
                errs += 1
                words = [startsym] + words + [stopsym]
                tags = [startsym] + tags + [stopsym]
                mytags = [startsym] + mytags + [stopsym]
                for i, (x, y, z) in enumerate(zip(words, tags, mytags)[1:], 1):
                    if y != z:
                        errwords += 1
                        model_0[y, x] += 1
                        model_0[z, x] -= 1
                        model_c[y, x] += c
                        model_c[z, x] -= c
                        features[y, x] = 1
                        features[z, x] = 1
                    if y!= z or tags[i-1] != mytags[i-1]:
                        model_0[tags[i-1], y] += 1
                        model_0[mytags[i-1], z] -= 1
                        model_c[tags[i-1], y] += c
                        model_c[mytags[i-1], z] -= c
                        features[tags[i-1], y] = 1
                        features[mytags[i-1], z] = 1
            c += 1
        for item in model_0:
            model[item] = model_0[item] - model_c[item] / c
        dev_err = test(devfile, dictionary, model)
        num_feature = sum(features[feature] for feature in features)
        print "epoch %d, updates %d, |w| = %d, train_err %2.f%%, dev_err %.2f%%" %(epoch, errs, num_feature, float(errwords)/whole*100, dev_err*100)
        train_avg_percep.append(float(errwords/whole)*100)
        dev_avg_percep.append(dev_err * 100)
    return train_avg_percep, dev_avg_percep


trainfile = 'train.txt.lower.unk'
devfile = 'dev.txt.lower.unk'
dictionary, _ = mle(trainfile)
#print dictionary
t = time.time()
train_avg_err, dev_avg_err = avg_perceptron(trainfile, devfile, dictionary, epochs = 5)
t2 = time.time()
print t2-t


epoch = [i for i in range(1, 6)]
plt.figure()
plt.plot(epoch, train_err, 'r-', label = 'Unaveraged Perceptron train_err')
plt.plot(epoch, dev_err, 'r--', label = 'Unaveraged Perceptron dev_err')
plt.plot(epoch, train_avg_err, 'b-', label = 'Averaged Perceptron train_err')
plt.plot(epoch, dev_avg_err, 'b--', label = 'Averaged Perceptron dev_err')
plt.legend(loc = 1)
plt.xlabel("Epoch", fontsize = 12)
plt.ylabel("Error (\%)", fontsize = 12)
plt.ylim([1, 14])
plt.show()


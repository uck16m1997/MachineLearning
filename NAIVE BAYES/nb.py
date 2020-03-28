import string

from nltk.stem.porter import PorterStemmer as Stemmer
import re
from os import listdir
from os.path import isfile, join
from math import log

stopwords = ["ourselves", "hers", "between", "yourself", "but", "again", "there", "about", "once", "during", "out",
             "very", "having", "with", "they", "own", "an", "be", "some", "for", "do", "its", "yours", "such", "into",
             "of", "most", "itself", "other", "off", "is", "s", "am", "or", "who", "as", "from", "him", "each", "the",
             "themselves", "until", "below", "are", "we", "these", "your", "his", "through", "don", "nor", "me", "were",
             "her", "more", "himself", "this", "down", "should", "our", "their", "while", 'above', 'both', 'up', 'to',
             'ours', 'had', 'she', 'all', 'no', 'when', 'at', 'any', 'before', 'them', 'same', 'and', 'been', 'have',
             'in', 'will', 'on', 'does', 'yourselves', 'then', 'that', 'because', 'what', 'over', 'why', 'so', 'can',
             'did', 'not', 'now', 'under', 'he', 'you', 'herself', 'has', 'just', 'where', 'too', 'only', 'myself',
             'which', 'those', 'i', 'after', 'few', 'whom', 't', 'being', 'if', 'theirs', 'my', 'against', 'a', 'by',
             'doing', 'it', 'how', 'further', 'was', 'here', 'than']
#stopwords = []

class NaiveBayes:
    def __init__(self):
        self.stemmer = Stemmer()
        self.di = {}
        self.priors = [0.0, 0.0]
        self.counts = [0, 0]

    def accuracy(self, data, real):
        pred = self.predict(data)
        correct = 0
        for i in range(len(pred)):
            if pred[i] == real[i]:
                correct += 1
        print("Accuracy is ", correct / len(data))

    def train(self, data):

        for t in data:
            self.priors[t[1]] += 1.0
            f = open(t[0], "rb+")
            # word = f.read().decode("latin-1")
            # word = word.split(" ")
            words = re.findall(r'\w[a-z]+', f.read().decode('latin-1').lower())
            for w in words:
                stemmed = self.stemmer.stem(w)

                if (stemmed in stopwords):
                    continue
                if (stemmed in self.di.keys()):
                    self.di[stemmed][t[1]] += 1
                    self.counts[t[1]]+=1

                else:
                    self.di[stemmed] = {0: 0, 1: 0}
                    self.counts[t[1]]+=1
                    self.di[stemmed][t[1]] += 1

        self.priors[0] = self.priors[0] / len(data)
        self.priors[1] = self.priors[1] / len(data)

        #self.removeFreqs()

    def removeFreqs(self):
        top3per = int(len(self.di) * 1 / 100)
        freqw = []
        smallest = [0, 0]
        for w in self.di:
            total = self.di[w][0] + self.di[w][1]
            if (len(freqw) < top3per):
                if (total < smallest[0]):
                    smallest[0] = total
                    smallest[1] = len(freqw) - 1
                freqw.append([w, total])
            elif (total < smallest[0]):
                freqw[total[1]] = w
                for i in range(len(freqw)):
                    if (freqw[i][1] < total[0]):
                        total[0] = freqw[i][1]
                        total[1] = i
        for w in freqw:
           stopwords.append(w)

        # print(freqw)

    def predict(self, data):
        pred = []
        k = 1
        #self.priors[0] = log(self.priors[0])
        #self.priors[1] = log(self.priors[1])
        for fn in data:
            probs = [0, 0]
            f = open(fn, "rb+")
            words = re.findall(r'\w[a-z]+', f.read().decode('latin-1').lower())
            for i in range(2):


                denom = (self.counts[i] + k * len(self.di))

                for w in words:
                    stemmed = self.stemmer.stem(w)
                    if (stemmed in stopwords):
                        continue
                    numerator = k*1
                    x = self.di.get(stemmed,0)
                    if(x!=0):
                        x=x[i]
                    numerator +=x
                    div = numerator / denom
                    probs[i] += log(div)


            probs[0] = probs[0]+log(self.priors[0])
            probs[1] = probs[1]+log(self.priors[1])


            if (probs[0] > probs[1]):
                pred.append(0)
            else:
                pred.append(1)
        return pred


def traindata():
    trainfiles = [(join("train/ham", f), 1) for f in listdir("train/ham") if isfile(join("train/ham", f))]
    for f in listdir("train/spam"):
        if (isfile(join("train/spam", f))):
            trainfiles.append((join("train/spam", f), 0))
    return trainfiles


# print(trainfiles)
def testdata():
    testfiles = [join("test/spam", f) for f in listdir("test/spam") if isfile(join("test/spam", f))]
    real = [0] * len(testfiles)
    for f in listdir("test/ham"):
        if(isfile(join("test/ham", f))):
            testfiles.append(join("test/ham", f))
            real.append(1)

    return testfiles, real


# testfiles.append((join("test/spam",f),0) for f in listdir("test/spam") if isfile(join("test/spam", f)))

# real.extend([0]*(len(testfiles)-len(real))


trainfiles = traindata()
nb = NaiveBayes()
nb.train(trainfiles)
testfiles, real = testdata()

pred = nb.predict(testfiles)
print(pred)

nb.accuracy(testfiles, real)

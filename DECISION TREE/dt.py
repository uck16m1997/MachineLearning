import csv
import sys

def main(trainset="training_set.csv",validset="validation_set.csv",testset="test_set.csv",toprint='yes',heur=1):
    train = []
    header = []
    with open("/Users/umutck/Desktop/Machine Learning/dataset 1/"+trainset) as csvfile:
        S = csv.reader(csvfile, delimiter=',')
        count = 0
        for row in S:
            if count == 0:
                header = row
                count += 1
                continue
            train.append([int(i) for i in row])

    dt = DecisonTree(h=heur)
    dt.train(train)
    if(toprint=='yes'):
        dt.printtree(header)

    test = []
    with open("/Users/umutck/Desktop/Machine Learning/dataset 1/"+testset) as csvfile:
        S = csv.reader(csvfile, delimiter=',')
        count = 0
        for row in S:
            if count == 0:
                header = row
                count += 1
                continue
            test.append([int(i) for i in row])

    validation = []
    with open("/Users/umutck/Desktop/Machine Learning/dataset 1/"+validset) as csvfile:
        S = csv.reader(csvfile, delimiter=',')
        count = 0
        for row in S:
            if count == 0:
                header = row
                count += 1
                continue
            validation.append([int(i) for i in row])

    print()
    print("Accuracy on training set is: ",dt.accuracy(train))

    print()
    print("Accuracy on test set is: ",dt.accuracy(test))

    print()
    print("Accuracy on validation set is: ",dt.accuracy(validation))

class node:
    def __init__(self):
        # self.left = None
        # self.right = None
        self.nodes = [0, 1]
        self.atr = None
        self.data = None


class DecisonTree:
    def __init__(self, s=[],h=1):
        self.root = node()
        self.root.data = s
        self.heur = h
        #self.buildtree(self.root)

    def train(self,s):
        self.root.data =s
        self.buildtree(self.root)

    def predict(self,s):
        res = []
        stack = []
        for row in s:
            stack.append(self.root)
            while(len(stack)>0):
                n = stack.pop()
                if(row[n.atr]==0):
                    if(n.nodes[0]!=1 and n.nodes[0]!=0  and  n.nodes[0].atr!=None):
                        stack.append(n.nodes[0])
                    elif(n.nodes[0]!=1  and n.nodes[0]!=0  and n.nodes[0].atr==None):

                        res.append(n.nodes[0].data[0][-1])
                elif(row[n.atr]==1):
                    if(n.nodes[1]!=1 and n.nodes[1]!=0 and n.nodes[1].atr!=None):
                        stack.append(n.nodes[1])
                    elif(n.nodes[1]!=1 and n.nodes[1]!=0  and n.nodes[1].atr==None):
                        res.append(n.nodes[1].data[0][-1])
        return res

    def accuracy(self,s):
        ans = self.predict(s)
        true = 0
        for i in range(len(ans)):
            if (ans[i] == s[i][-1]):
                true += 1

        #print(true)
        #print(len(ans))
        return(true/len(ans)*100)


    def vi(self, s, atr=-1):

        k = len(s)
        k0 = 0
        k1 = 0
        for data in s:
            c = (data[atr])
            if c == 0:
                k0 += 1
        k1 = k - k0

        return (k0 * k1) / (k ** 2)

    def prob(self, atr, x, s):
        k = 0
        for data in s:
            if data[atr] == x:
                k += 1
        return k / len(s)

    def gain(self, atr, s):
        bigv = self.vi(self.root.data)
        ans = 0
        i = 0
        while i < 2:
            sx = []
            for r in s:
                if r[atr] == i:
                    sx.append(r)
            if len(sx) == 0:
                return bigv
            prob = self.prob(atr, i, s)
            vi = self.vi(sx)
            ans += prob * vi
            i += 1
        return bigv - ans

    def mini(self, sx, locked=[]):
        mindex = -1
        maxGain = 0
        if (self.heur != 1):
            maxGain = 1
        for atr in range(len(sx[0]) - 1):
            if (atr in locked):
                continue
            gain = 0
            if(self.heur==1):
                gain = self.gain(atr, sx)
                if (gain == 1):
                    mindex = atr
                    maxGain = 1
                    return mindex, maxGain
                if (gain > maxGain):
                    mindex = atr
                    maxGain = gain
            else:
                gain = self.vi(sx, atr)

                if (gain == 0):
                    mindex = atr
                    maxGain = 1
                    return mindex, maxGain
                if (gain < maxGain):
                    mindex = atr
                    maxGain = gain



        return mindex, maxGain

    def buildtree(self, crnode, locked=[]):
        mindex, maxGain = self.mini(crnode.data, locked)
        if(mindex==-1):
            return locked

        locked.append(mindex)
        crnode.atr = mindex
        i = 0
        while (i < 2):
            sx = []
            for r in crnode.data:
                if (r[mindex] == i):
                    sx.append(r)
            if(len(sx)==0 or len(sx)==len(crnode.data)):
                crnode.atr=None
                cpy = locked[:]
                self.buildtree(crnode, cpy)
                locked.pop()
                return locked

            if (i == 0):
                crnode.nodes[i] = node()
                #crnode.nodes[i].atr=mindex
                crnode.nodes[i].data = sx
                total = 0
                for row in sx:
                    total+=row[-1]
                if(total!=0 and total!=len(sx)):
                    cpy = locked[:]
                    self.buildtree(crnode.nodes[i],cpy)
            else:
                crnode.nodes[i] = node()
                # crnode.nodes[i].atr=mindex
                crnode.nodes[i].data = sx
                total = 0
                for row in sx:
                    total+=row[-1]
                if(total!=0 and total!=len(sx)):
                    cpy = locked[:]
                    self.buildtree(crnode.nodes[i],cpy)
            i += 1
        return locked

    def printtree(self,header):
        stack = []

        stack.append((self.root,0,1))
        stack.append((self.root,0,0))


        while (len(stack) > 0):
            n = stack.pop()
            for i in range(n[1]):
                print("|",end='')
            print(header[n[0].atr],end= " = "+str(n[2])+" : ")

            if(n[0].nodes[n[2]]!=0 and n[0].nodes[n[2]]!=1 and n[0].nodes[n[2]].atr!=None):
                    #print(" 1 :", end="")
                    stack.append((n[0].nodes[n[2]], n[1] + 1, 1))
                    stack.append((n[0].nodes[n[2]], n[1] + 1, 0))

            elif(n[0].nodes[n[2]]!=0 and n[0].nodes[n[2]]!=1 and n[0].nodes[n[2]].atr==None):

                print(n[0].nodes[n[2]].data[0][-1])
                continue

            print()


if __name__ == "__main__":
    if(len(sys.argv)!=6):
        print("Wrong number of arguments")
        exit()
    if(int(sys.argv[5])>2 or int(sys.argv[5])<1):
        print("heuristic can only be 1 or 2")
        exit()
    main(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4],int(sys.argv[5]))




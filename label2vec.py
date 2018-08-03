import sys
import gensim

labels = [[]]

with open(sys.argv[1]) as file:

    for line in file:

        line = line.split()

        if line: labels[-1].append(line[7])
        
        else: labels.append([])

with open("train_labels.txt", 'w') as file:

    for label in labels:

        file.write(" ".join(label))
        file.write("\n")

labels = gensim.models.word2vec.LineSentence("train_labels.txt")
label2vec = gensim.models.word2vec.Word2Vec(labels)
label2vec.save("label2vec.model")

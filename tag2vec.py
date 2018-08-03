import sys
import gensim

tags = [[]]

with open(sys.argv[1]) as file:

    for line in file:

        line = line.split()

        if line: tags[-1].append(line[3])
        
        else: tags.append([])

with open("train_tags.txt", 'w') as file:

    for tag in tags:

        file.write(" ".join(tag))
        file.write("\n")

tags = gensim.models.word2vec.LineSentence("train_tags.txt")
tag2vec = gensim.models.word2vec.Word2Vec(tags)
tag2vec.save("tag2vec.model")

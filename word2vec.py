import gensim

tags = [[]]
labels = [[]]

with open("train.conll") as file:

    for line in file:

        line = line.split()

        if line:
            tags[-1].append(line[3])
            labels[-1].append(line[7])
        
        else:

            tags.append([])
            labels.append([])

with open("train_tags.txt", 'w') as file:

    for tag in tags:

        file.write(" ".join(tag))
        file.write("\n")

with open("train_labels.txt", 'w') as file:

    for label in labels:
        
        file.write(" ".join(label))
        file.write("\n")

tags = gensim.models.word2vec.LineSentence("train_tags.txt")
tag2vec = gensim.models.word2vec.Word2Vec(tags)
tag2vec.save("tag2vec.model")

labels = gensim.models.word2vec.LineSentence("train_labels.txt")
label2vec = gensim.models.Word2Vec(labels)
label2vec.save("label2vec.model")
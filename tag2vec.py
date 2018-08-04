import sys
import gensim

sentences = [[]]

with open(sys.argv[1]) as file:

    for line in file:

        line = line.split()

        if line: sentences[-1].append(line[3])
        
        else: sentences.append([])

with open("train_tags.txt", 'w') as file:

    for sentence in sentences:

        file.write(" ".join(sentence))
        file.write("\n")

sentences = gensim.models.word2vec.LineSentence("train_tags.txt")
tag2vec = gensim.models.word2vec.Word2Vec(sentences=sentences, min_count=1)
tag2vec.save("tag2vec.model")

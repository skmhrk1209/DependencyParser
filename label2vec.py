import sys
import gensim

sentences = [[]]

with open(sys.argv[1]) as file:

    for line in file:

        line = line.split()

        if line: sentences[-1].append(line[7])
        
        else: sentences.append([])

with open("train_labels.txt", 'w') as file:

    for sentence in sentences:

        file.write(" ".join(sentence))
        file.write("\n")

sentences = gensim.models.word2vec.LineSentence("train_labels.txt")
label2vec = gensim.models.word2vec.Word2Vec(sentences=sentences, min_count=1)
label2vec.save("label2vec.model")

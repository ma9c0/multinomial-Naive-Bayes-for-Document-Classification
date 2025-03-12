from utils import load_training_set, load_test_set
from collections import Counter
from math import log
import matplotlib.pyplot as plt

if __name__ == '__main__':
    percentage_positive_instances_train = 0.2
    percentage_negative_instances_train = 0.2

    percentage_positive_instances_test = 0.2
    percentage_negative_instances_test = 0.2

    (pos_train, neg_train, vocab) = load_training_set(percentage_positive_instances_train, percentage_negative_instances_train)
    (pos_test, neg_test) = load_test_set(percentage_positive_instances_test, percentage_negative_instances_test)

    print("Number of positive training instances:", len(pos_train))
    print("Number of negative training instances:", len(neg_train))
    print("Number of positive test instances:", len(pos_test))
    print("Number of negative test instances:", len(neg_test))
    
    def concat_words(set):
        bag = []
        for instance in set:
            for words in instance:
                bag.extend(words)
        return bag
    
    pos_train_bag = concat_words(pos_train)
    pos_train_count = Counter(pos_train_bag)
    neg_train_bag = concat_words(neg_train)
    neg_train_count = Counter(neg_train_bag)
    train_len = len(pos_train) + len(neg_train)
    vocab_count = Counter(vocab)
    
    #Q1:
    def class_of_doc_standard(doc):
        def class_prob(doc, class_bag, count):
            Pr_wk_yi = 1
            for word in doc:
                Pr_wk_yi *= (count[word] / len(class_bag))
            return Pr_wk_yi
        
        pos_prob = (len(pos_train) / train_len) * class_prob(doc, pos_train_bag, pos_train_count)
        neg_prob = (len(neg_train) / train_len) * class_prob(doc, neg_train_bag, neg_train_count)
        
        return 1 if pos_prob > neg_prob else 0
    
    TP, FP, TN, FN = 0, 0, 0, 0
    
    for ins in pos_test:
        if class_of_doc_standard(ins) == 1:
            TP += 1
        else:
            FN += 1
    
    for ins in neg_test:
        if class_of_doc_standard(ins) == 0:
            TN += 1
        else:
            FP += 1
            
    print("Q1:")
    print("Accuracy: ", (TP + TN) / (TP + FP + TN + FN))
    print("Precision: ", TP / (TP + FP) if (TP + FP) > 0 else 0)
    print("Recall: ", TP / (TP + FN) if (TP + FN) > 0 else 0)
    
    confusion_matrix = [["Confusion Matrix", "Predicted = Pos", "Neg"],
                       ["Actual = Pos", "TP:" + str(TP), "FN:" + str(FN)],
                       ["Neg", "FP: " + str(FP), "TN: " + str(TN)]]
    
    for row in confusion_matrix:
        print(row)
    
    # Q2:
    def class_of_doc(doc, alpha):
        def log_wk_yi(word, label_count, label_bag):
            num = label_count.get(word, 0) + alpha
            dom = sum(label_count.values()) + alpha * len(vocab)
            return log(num / dom)
        
        logp_pos = log(len(pos_train) / train_len)
        logp_neg = log(len(neg_train) / train_len)
        
        ll_pos = sum(log_wk_yi(word, pos_train_count, pos_train_bag) for word in doc)
        ll_neg = sum(log_wk_yi(word, neg_train_count, neg_train_bag) for word in doc)
        
        lp_pos = logp_pos + ll_pos
        lp_neg = logp_neg + ll_neg
        
        return 1 if lp_pos > lp_neg else 0
    
    a = 1.0
    TP, FP, TN, FN = 0, 0, 0, 0

    for ins in pos_test:
        if class_of_doc(ins, a) == 1:
            TP += 1
        else:
            FN += 1
    
    for ins in neg_test:
        if class_of_doc(ins, a) == 0:
            TN += 1
        else:
            FP += 1

    print("\nQ2:")
    print("Accuracy: ", (TP + TN) / (TP + FP + TN + FN))
    print("Precision: ", TP / (TP + FP) if (TP + FP) > 0 else 0)
    print("Recall: ", TP / (TP + FN) if (TP + FN) > 0 else 0)
    
    confusion_matrix = [["Confusion Matrix", "Predicted = Pos", "Neg"],
                       ["Actual = Pos", "TP:" + str(TP), "FN:" + str(FN)],
                       ["Neg", "FP: " + str(FP), "TN: " + str(TN)]]
    
    for row in confusion_matrix:
        print(row)
    
    alphas = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
    accuracies = []

    for a in alphas:
        correct = 0
        for ins in pos_test:
            if class_of_doc(ins, a) == 1:
                correct += 1
        for ins in neg_test:
            if class_of_doc(ins, a) == 0:
                correct += 1
        total_tests = len(pos_test) + len(neg_test)
        acc = correct / total_tests
        accuracies.append(acc)
        
    print(accuracies)
        
    plt.plot(alphas, accuracies, marker='o')
    plt.xscale('log')
    plt.xlabel('Alpha (log scale)')
    plt.ylabel('Accuracy')
    plt.title('Naive Bayes Accuracy vs. Alpha')
    plt.show()
#!/usr/bin/env python
#
# Example consumer of queue messages
#
# pip3 install -r requirements.txt
#
import argparse
import sys
import os
import pika
import signal
import json

from math import log
import re
import math
import operator

# decision tree learning
def get_words(doc):
    """Поделба на документот на зборови. Стрингот се дели на зборови според
    празните места и интерпукциските знаци

    :param doc: документ
    :type doc: str
    :return: множество со зборовите кои се појавуваат во дадениот документ
    :rtype: set(str)
    """
    # подели го документот на зборови и конвертирај ги во мали букви
    # па потоа стави ги во резултатот ако нивната должина е >2 и <20
    words = set()
    for word in re.split('\\W+', doc):
        if 2 < len(word) < 20:
            words.add(word.lower())
    return words


class DocumentClassifier:
    def __init__(self, get_features):
        # број на парови атрибут/категорија (feature/category)
        self.feature_counts_per_category = {}
        # број на документи во секоја категорија
        self.category_counts = {}
        # функција за добивање на атрибутите (зборовите) во документот
        self.get_features = get_features

    def increment_feature_counts_per_category(self, current_feature, current_category):
        """Зголемување на бројот на парови атрибут/категорија

        :param current_feature: даден атрибут
        :param current_category: дадена категорија
        :return: None
        """
        self.feature_counts_per_category.setdefault(current_feature, {})
        self.feature_counts_per_category[current_feature].setdefault(current_category, 0)
        self.feature_counts_per_category[current_feature][current_category] += 1

    def increment_category_counts(self, cat):
        """Зголемување на бројот на предмети (документи) во категорија

        :param cat: категорија
        :return: None
        """
        self.category_counts.setdefault(cat, 0)
        self.category_counts[cat] += 1

    def get_feature_counts_per_category(self, current_feature, current_category):
        """Добивање на бројот колку пати одреден атрибут се има појавено во
        одредена категорија

        :param current_feature: атрибут
        :param current_category: категорија
        :return: None
        """
        if current_feature in self.feature_counts_per_category \
                and current_category in self.feature_counts_per_category[current_feature]:
            return float(self.feature_counts_per_category[current_feature][current_category])
        return 0.0

    def get_category_count(self, current_category):
        """Добивање на бројот на предмети (документи) во категорија

        :param current_category: категорија
        :return: број на предмети (документи)
        """
        if current_category in self.category_counts:
            return float(self.category_counts[current_category])
        return 0

    def get_total_count(self):
        """Добивање на вкупниот број на предмети"""
        return sum(self.category_counts.values())

    def categories(self):
        """Добивање на листа на сите категории"""
        return self.category_counts.keys()

    def train(self, item, current_category):
        """Тренирање на класификаторот. Новиот предмет (документ)

        :param item: нов предмет (документ)
        :param current_category: категорија
        :return: None
        """
        # Се земаат атрибутите (зборовите) во предметот (документот)
        features = self.get_features(item)
        # Се зголемува бројот на секој атрибут во оваа категорија
        for current_feature in features:
            self.increment_feature_counts_per_category(current_feature, current_category)

        # Се зголемува бројот на предмети (документи) во оваа категорија
        self.increment_category_counts(current_category)

    def get_feature_per_category_probability(self, current_feature, current_category):
        """Веројатноста е вкупниот број на пати кога даден атрибут f (збор) се појавил во
        дадена категорија поделено со вкупниот број на предмети (документи) во категоријата

        :param current_feature: атрибут
        :param current_category: карактеристика
        :return: веројатност на појавување
        """
        if self.get_category_count(current_category) == 0:
            return 0
        return self.get_feature_counts_per_category(current_feature, current_category) \
               / self.get_category_count(current_category)

    def weighted_probability(self, current_feature, current_category, prf, weight=1.0, ap=0.5):
        """Пресметка на тежински усогласената веројатност

        :param current_feature: атрибут
        :param current_category: категорија
        :param prf: функција за пресметување на основната веројатност
        :param weight: тежина
        :param ap: претпоставена веројатност
        :return: тежински усогласена веројатност
        """
        # Пресметај ја основната веројатност
        basic_prob = prf(current_feature, current_category)
        # Изброј колку пати се има појавено овој атрибут (збор) во сите категории
        totals = sum([self.get_feature_counts_per_category(current_feature, currentCategory) for currentCategory in
                      self.categories()])
        # Пресметај ја тежински усредената веројатност
        bp = ((weight * ap) + (totals * basic_prob)) / (weight + totals)
        return bp


class NaiveBayes(DocumentClassifier):
    def __init__(self, get_features):
        super().__init__(get_features)
        self.thresholds = {}

    def set_threshold(self, current_category, threshold):
        """Поставување на праг на одлучување за категорија

        :param current_category: категорија
        :param threshold: праг на одлучување
        :return: None
        """
        self.thresholds[current_category] = threshold

    def get_threshold(self, current_category):
        """Добивање на прагот на одлучување за дадена класа

        :param current_category: категорија
        :return: праг на одлучување за дадената категорија
        """
        if current_category not in self.thresholds:
            return 1.0
        return self.thresholds[current_category]

    def calculate_document_probability_in_class(self, item, current_category):
        """Ја враќа веројатноста на документот да е од класата current_category
        (current_category е однапред позната)

        :param item: документ
        :param current_category: категорија
        :return:
        """
        # земи ги зборовите од документот item
        features = self.get_features(item)
        # помножи ги веројатностите на сите зборови
        p = 1
        for current_feature in features:
            p *= self.weighted_probability(current_feature, current_category,
                                           self.get_feature_per_category_probability)

        return p

    def get_category_probability_for_document(self, item, current_category):
        """Ја враќа веројатноста на класата ако е познат документот

        :param item: документ
        :param current_category: категорија
        :return: веројатност за документот во категорија
        """
        cat_prob = self.get_category_count(current_category) / self.get_total_count()
        calculate_document_probability_in_class = self.calculate_document_probability_in_class(item, current_category)
        # Bayes Theorem
        return calculate_document_probability_in_class * cat_prob / (1.0 / self.get_total_count())

    def classify_document(self, item, default=None):
        """Класифицирање на документ

        :param item: документ
        :param default: подразбирана (default) класа
        :return:
        """
        probs = {}
        # најди ја категоријата (класата) со најголема веројатност
        max = 0.0
        for cat in self.categories():
            probs[cat] = self.get_category_probability_for_document(item, cat)
            if probs[cat] > max:
                max = probs[cat]
                best = cat

        # провери дали веројатноста е поголема од threshold*next best (следна најдобра)
        for cat in probs:
            if cat == best:
                continue
            if probs[cat] * self.get_threshold(best) > probs[best]: return default

        return best


def unique_counts(rows):
    """Креирај броење на можни резултати (последната колона
    во секоја редица е класата)

    :param rows: dataset
    :type rows: list
    :return: dictionary of possible classes as keys and count
             as values
    :rtype: dict
    """
    results = {}
    for row in rows:
        # Клацата е последната колона
        r = row[len(row) - 1]
        if r not in results:
            results[r] = 0
        results[r] += 1
    return results


def gini_impurity(rows):
    """Probability that a randomly placed item will
    be in the wrong category

    :param rows: dataset
    :type rows: list
    :return: Gini impurity
    :rtype: float
    """
    total = len(rows)
    counts = unique_counts(rows)
    imp = 0
    for k1 in counts:
        p1 = float(counts[k1]) / total
        for k2 in counts:
            if k1 == k2:
                continue
            p2 = float(counts[k2]) / total
            imp += p1 * p2
    return imp


def entropy(rows):
    """Ентропијата е сума од p(x)log(p(x)) за сите
    можни резултати

    :param rows: податочно множество
    :type rows: list
    :return: вредност за ентропијата
    :rtype: float
    """
    log2 = lambda x: log(x) / log(2)
    results = unique_counts(rows)
    # Пресметка на ентропијата
    ent = 0.0
    for r in results.keys():
        p = float(results[r]) / len(rows)
        ent = ent - p * log2(p)
    return ent


class DecisionNode:
    def __init__(self, col=-1, value=None, results=None, tb=None, fb=None):
        """
        :param col: индексот на колоната (атрибутот) од тренинг множеството
                    која се претставува со оваа инстанца т.е. со овој јазол
        :type col: int
        :param value: вредноста на јазолот според кој се дели дрвото
        :param results: резултати за тековната гранка, вредност (различна
                        од None) само кај јазлите-листови во кои се донесува
                        одлуката.
        :type results: dict
        :param tb: гранка која се дели од тековниот јазол кога вредноста е
                   еднаква на value
        :type tb: DecisionNode
        :param fb: гранка која се дели од тековниот јазол кога вредноста е
                   различна од value
        :type fb: DecisionNode
        """
        self.col = col
        self.value = value
        self.results = results
        self.tb = tb
        self.fb = fb


def compare_numerical(row, column, value):
    """Споредба на вредноста од редицата на посакуваната колона со
    зададена нумеричка вредност

    :param row: дадена редица во податочното множество
    :type row: list
    :param column: индекс на колоната (атрибутот) од тренирачкото множество
    :type column: int
    :param value: вредност на јазелот во согласност со кој се прави
                  поделбата во дрвото
    :type value: int or float
    :return: True ако редицата >= value, инаку False
    :rtype: bool
    """
    return row[column] >= value


def compare_nominal(row, column, value):
    """Споредба на вредноста од редицата на посакуваната колона со
    зададена номинална вредност

    :param row: дадена редица во податочното множество
    :type row: list
    :param column: индекс на колоната (атрибутот) од тренирачкото множество
    :type column: int
    :param value: вредност на јазелот во согласност со кој се прави
                  поделбата во дрвото
    :type value: str
    :return: True ако редицата == value, инаку False
    :rtype: bool
    """
    return row[column] == value


def divide_set(rows, column, value):
    """Поделба на множеството според одредена колона. Може да се справи
    со нумерички или номинални вредности.

    :param rows: тренирачко множество
    :type rows: list(list)
    :param column: индекс на колоната (атрибутот) од тренирачкото множество
    :type column: int
    :param value: вредност на јазелот во зависност со кој се прави поделбата
                  во дрвото за конкретната гранка
    :type value: int or float or str
    :return: поделени подмножества
    :rtype: list, list
    """
    # Направи функција која ни кажува дали редицата е во
    # првата група (True) или втората група (False)
    if isinstance(value, int) or isinstance(value, float):
        # ако вредноста за споредба е од тип int или float
        split_function = compare_numerical
    else:
        # ако вредноста за споредба е од друг тип (string)
        split_function = compare_nominal

    # Подели ги редиците во две подмножества и врати ги
    # за секој ред за кој split_function враќа True
    set1 = [row for row in rows if
            split_function(row, column, value)]
    # set1 = []
    # for row in rows:
    #     if not split_function(row, column, value):
    #         set1.append(row)
    # за секој ред за кој split_function враќа False
    set2 = [row for row in rows if
            not split_function(row, column, value)]
    return set1, set2


def build_tree(rows, scoref=entropy):
    if len(rows) == 0:
        return DecisionNode()
    current_score = scoref(rows)

    # променливи со кои следиме кој критериум е најдобар
    best_gain = 0.0
    best_criteria = None
    best_sets = None


    column_count = len(rows[0]) - 1
    for col in range(0, column_count):
        # за секоја колона (col се движи во интервалот од 0 до
        # column_count - 1)
        # Следниов циклус е за генерирање на речник од различни
        # вредности во оваа колона
        column_values = {}
        for row in rows:
            column_values[row[col]] = 1
        # за секоја редица се зема вредноста во оваа колона и се
        # поставува како клуч во column_values
        for value in column_values.keys():
            (set1, set2) = divide_set(rows, col, value)

            # Информациона добивка
            p = float(len(set1)) / len(rows)
            gain = current_score - p * scoref(set1) - (1 - p) * scoref(set2)
            if gain > best_gain and len(set1) > 0 and len(set2) > 0:
                best_gain = gain
                best_criteria = (col, value)
                best_sets = (set1, set2)

    # Креирај ги подгранките
    if best_gain > 0:
        true_branch = build_tree(best_sets[0], scoref)
        false_branch = build_tree(best_sets[1], scoref)
        return DecisionNode(col=best_criteria[0], value=best_criteria[1],
                            tb=true_branch, fb=false_branch)
    else:
        return DecisionNode(results=unique_counts(rows))


def print_tree(tree, indent=''):
    # Дали е ова лист јазел?
    if tree.results:
        print(str(tree.results))
    else:
        # Се печати условот
        print(str(tree.col) + ':' + str(tree.value) + '? ')
        # Се печатат True гранките, па False гранките
        print(indent + 'T->', end='')
        print_tree(tree.tb, indent + '  ')
        print(indent + 'F->', end='')
        print_tree(tree.fb, indent + '  ')


def classify(observation, tree):
    if tree.results:
        return tree.results
    else:
        value = observation[tree.col]
        if isinstance(value, int) or isinstance(value, float):
            compare = compare_numerical
        else:
            compare = compare_nominal

        if compare(observation, tree.col, tree.value):
            branch = tree.tb
        else:
            branch = tree.fb

        return classify(observation, branch)


# tf idf

def get_words(doc):
    """Поделба на документот на зборови. Стрингот се дели на зборови според
    празните места и интерпукциските знаци

    :param doc: документ
    :type doc: str
    :return: множество со зборовите кои се појавуваат во дадениот документ
    :rtype: set(str)
    """
    # подели го документот на зборови и конвертирај ги во мали букви
    # па потоа стави ги во резултатот ако нивната должина е >2 и <20
    words = list()
    for word in re.split('\\W+', doc):
        if 2 < len(word) < 20:
            words.append(word.lower())
    return words


def get_vocabulary(documents):
    """Враќа множество од сите зборови кои се појавуваат во документите.

    :param documents: листа со документи
    :type documents: list(str)
    :return: множество зборови
    :rtype: set(str)
    """
    vocab = set()
    for doc_text in documents:
        words = get_words(doc_text)
        words_set = set(words)
        vocab.update(words_set)
    return sorted(vocab)


def cosine(v1, v2):
    """Ја враќа косинусната сличност помеѓу два вектори v1 и v2.

    :param v1: вектор1
    :type v1: list(float)
    :param v2: вектор2
    :type v2: list(float)
    :return: сличност помеѓу вектор и вектор2
    :rtype: float
    """
    sumxx, sumxy, sumyy = 0, 0, 0
    for i in range(len(v1)):
        x = v1[i]
        y = v2[i]
        sumxx += x * x
        sumyy += y * y
        sumxy += x * y
    return sumxy / math.sqrt(sumxx * sumyy)


def pearson(v1, v2):
    """ Го враќа коефициентот на Пирсонова корелација помеѓу два вектори v1 и v2.

    :param v1: вектор1
     :type v1: list(float)
    :param v2: вектор2
    :type v2: list(float)
    :return: сличност помеѓу вектор и вектор2
    :rtype: float
    """
    sum1 = 0
    sum2 = 0
    sum1Sq = 0
    sum2Sq = 0
    pSum = 0
    n = len(v1)
    for i in range(n):
        x1 = v1[i]
        x2 = v2[i]
        sum1 += x1
        sum1Sq += x1 ** 2
        sum2 += x2
        sum2Sq += x2 ** 2
        pSum += x1 * x2
    num = pSum - (sum1 * sum2 / n)
    den = math.sqrt((sum1Sq - sum1 ** 2 / n) * (sum2Sq - sum2 ** 2 / n))
    if den == 0: return 0
    r = num / den
    return r


def calculate_document_frequencies(documents):
    """Враќа речник со број на појавување на зборовите.

    :param documents: листа со документи
    :type documents: list(str)
    :return: речник со број на појавување на зборовите
    :rtype: dict(str, int)
    """
    df = {}
    documents_words = []
    for doc_text in documents:
        words = get_words(doc_text)
        documents_words.append(words)
        words_set = set(words)
        for word in words_set:
            df.setdefault(word, 0)
            df[word] += 1
    return df


def calc_vector(cur_tf_idf, vocab):
    """Пресметува tf-idf вектор за даден документ од дадениот вокабулар.

    :param cur_tf_idf: речник со tf-idf тежини
    :type cur_tf_idf: dict(str, float)
    :param vocab: множество од сите зборови кои се појавуваат во барем еден документ
    :type vocab: set(str)
    :return: tf-idf вектор за дадениот документ
    """
    vec = []
    for word in vocab:
        tf_idf = cur_tf_idf.get(word, 0)
        vec.append(tf_idf)
    return vec


def process_document(doc, df, N, vocab):
    """Пресметува tf-idf за даден документ.

    :param doc: документ
    :type doc: str
    :param df: речник со фреквенции на зборовите во дадениот документ
    :type df: dict(str, int)
    :param N: вкупен број на документи
    :param vocab: множество од сите зборови кои се појавуваат во барем еден документ
    :type vocab: set(str)
    :return: tf-idf вектор за дадениот документ
    """
    if isinstance(doc, str):
        words = get_words(doc)
    else:
        words = doc
    idf = {}
    for word, cdf in df.items():
        idf[word] = math.log(N / cdf)
    f = {}  # колку пати се јавува секој збор во овој документ
    for word in words:
        f.setdefault(word, 0)
        f[word] += 1
    max_f = max(f.values())  # колку пати се појавува најчестиот збор во овој документ
    tf_idf = {}
    for word, cnt in f.items():
        ctf = cnt * 1.0 / max_f
        tf_idf[word] = ctf * idf.get(word, 0)
    vec = calc_vector(tf_idf, vocab)
    return vec


def rank_documents(doc, documents, sim_func=cosine):
    """Враќа најслични документи со дадениот документ.

    :param doc: документ
    :type doc: str
    :param documents: листа со документи
    :type documents: list(str)
    :param sim_func: функција за сличност
    :return: листа со најслични документи
    """
    df = calculate_document_frequencies(documents)
    N = len(documents)
    vocab = get_vocabulary(documents)
    doc_vectors = []
    for document in documents:
        vec = process_document(document, df, N, vocab)
        doc_vectors.append(vec)
    query_vec = process_document(doc, df, N, vocab)
    similarities = []
    for i, doc_vec in enumerate(doc_vectors):
        dist = sim_func(query_vec, doc_vec)
        similarities.append((dist, i))
    similarities.sort(reverse=True)
    return similarities


def create_dataset(documents, labels):
    """Формира податочно множество со tf-idf тежини и класи, соодветно за класификација со дрва на одлука.

    :param documents: листа со документи
    :type documents: list(str)
    :param labels: листа со класи
    :type labels: list
    :return: податочно множество со tf-idf тежини и класи, речник со френвенции на појавување на зборовите,
            број на документи во множеството, вокабулар од даденото множество на аборови
    :rtype: list(list), dict(str, int), int, set(word)
    """
    dataset = []
    doc_vectors = []
    df = calculate_document_frequencies(documents)
    N = len(documents)
    vocab = get_vocabulary(documents)
    for document in documents:
        vec = process_document(document, df, N, vocab)
        doc_vectors.append(vec)
    for doc_vec, label in zip(doc_vectors, labels):
        doc_vec.append(label)
        dataset.append(doc_vec)
    return dataset, df, N, vocab

def queue_callback(channel, method, properties, body):
    print("message received")
    decoded_body = body.decode('UTF-8')
    json_body = json.loads(decoded_body)
    file_contents_raw = json_body['data']['command']

    print(file_contents_raw)

    sample = file_contents_raw

    try:
        from sklearn.datasets import fetch_20newsgroups
    except ImportError:
        raise ImportError("Install sklearn to use the NewsGroupDataset")

    train = fetch_20newsgroups(subset='train')
    #test = fetch_20newsgroups(subset='test')
    train_data = [(' '.join(d.split()), str(t)) for d, t in zip(train['data'], train['target'])]

    #print(train_data)

    documents = [item[0] for item in train_data]
    labels = [item[1] for item in train_data]

    #print("documents")
    #print(documents)

    #("labels")
    #print(labels)

    dataset, df, N, vocab = create_dataset(documents, labels)

    sample_vector = process_document(sample, df, N, vocab)

    tree = build_tree(dataset)

    tree_prediction = max(classify(sample_vector, tree).items(), key=operator.itemgetter(1))[0]

    print("prediction is")
    print(tree_prediction)

def signal_handler(signal,frame):
  print("\nCTRL-C handler, cleaning up rabbitmq connection and quitting")
  connection.close()
  sys.exit(0)


example_usage = '''====EXAMPLE USAGE=====
Connect to remote rabbitmq host
--user=guest --password=guest --host=rabbitmq
Specify exchange and queue name
--exchange=default --queue=default
'''

ap = argparse.ArgumentParser(description="RabbitMQ producer",
                             epilog=example_usage,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('--user',default="guest",help="username e.g. 'guest'")
ap.add_argument('--password',default="guest",help="password e.g. 'pass'")
ap.add_argument('--host',default="rabbitmq",help="rabbitMQ host, defaults to localhost")
ap.add_argument('--port',type=int,default=5672,help="rabbitMQ port, defaults to 5672")
ap.add_argument('--exchange',default="",help="name of exchange to use, empty means default")
ap.add_argument('--queue',default="default",help="name of default queue, defaults to 'testqueue'")
ap.add_argument('--routing-key',default="default",help="routing key, defaults to 'testqueue'")
ap.add_argument('--body',default="my test!",help="body of message, defaults to 'mytest!'")
ap.add_argument('--vhost',default="/",help="body of message, defaults to 'mytest!'")
args = ap.parse_args()


# connect to RabbitMQ
credentials = pika.PlainCredentials(args.user, args.password)
connection = pika.BlockingConnection(pika.ConnectionParameters(args.host, args.port, '/', credentials))
channel = connection.channel()

channel.basic_consume(queue=args.queue, on_message_callback=queue_callback, auto_ack=True)

# capture CTRL-C
signal.signal(signal.SIGINT, signal_handler)

print("Waiting for messages, CTRL-C to quit...")
print("")
channel.start_consuming()
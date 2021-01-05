def ml(message):
    import pandas as pd

    dataset = pd.read_csv('./dataset.csv', 
                        header=None, 
                        names=["Comment", "Movie" ,"Bool"])

    dataset = dataset.tail(-1)
    dataset = dataset.drop("Movie", 1)

    dataset = dataset.sample(frac=.01, random_state=1).reset_index(drop=True)
    ########################################################################
    def map_points(x):
        x = float(x.replace(',', '.'))
        if x < 2.5:
            return False
        else:
            return True
    def clean_values(x):
        x = x.replace('\n', ' ')
        x = x.replace('\W', ' ')
        x = x.lower()
        return x

    dataset['Bool'] = dataset['Bool'].apply(map_points)
    dataset['Comment'] = dataset['Comment'].apply(clean_values)
    dataset['Comment'] = dataset['Comment'].str.split()
    ########################################################################
    vocabulary = []
    for comment in dataset['Comment']:
        for word in comment:
            vocabulary.append(word)
    # zayıflıklardan biri datasetin temizlenmesine çok bağlı
    vocabulary = list(set(vocabulary))

    # datanın temizlenmesi ve sözlüğün oluşması burada bitiyor
    ########################################################################
    # frekans tablosu oluşturuyoruz
    word_counts_per_comment = {unique_word: [0] * len(dataset['Comment']) for unique_word in vocabulary}

    for index, comment in enumerate(dataset['Comment']):
        for word in comment:
            word_counts_per_comment[word][index] += 1

    word_counts = pd.DataFrame(word_counts_per_comment)
    ########################################################################
    dataset_joined = pd.concat([dataset, word_counts], axis=1)
    ########################################################################
    import numpy as np

    positive_values = dataset_joined[dataset_joined['Bool'] == np.bool_(True)]
    negative_values = dataset_joined[dataset_joined['Bool'] == np.bool_(False)]

    positive_percentage = positive_values.shape[0] / len(dataset_joined)
    negative_percentage = negative_values.shape[0] / len(dataset_joined)

    n_words_per_positive_message = positive_values['Comment'].apply(len)
    n_positive = n_words_per_positive_message.sum()

    n_words_per_negative_message = negative_values['Comment'].apply(len)
    n_negative = n_words_per_negative_message.sum()

    n_vocabulary = len(vocabulary)

    alpha = 1 # laplace smoothing
    ########################################################################
    parameters_positive = {unique_word:0 for unique_word in vocabulary}
    parameters_negative = {unique_word:0 for unique_word in vocabulary}

    for word in vocabulary:
        n_word_given_positive = positive_values[word].sum()
        p_word_given_positive = (n_word_given_positive + alpha) / (n_positive + alpha*n_vocabulary)
        parameters_positive[word] = p_word_given_positive

        n_word_given_negative = negative_values[word].sum()
        p_word_given_negative = (n_word_given_negative + alpha) / (n_negative + alpha*n_vocabulary)
        parameters_negative[word] = p_word_given_negative
    ########################################################################
    import re
    import nltk
    from stop_words import stop_words

    WPT = nltk.WordPunctTokenizer()
    # Yukarıdaki rowları normalize etme işlemine de bu method uygulanabilir
    # bir ara implente et.
    def norm_doc(single_doc):
        # TR: Dokümandan belirlenen özel karakterleri ve sayıları at
        # EN: Remove special characters and numbers
        single_doc = re.sub(" \d+", " ", single_doc)
        pattern = r"[{}]".format(",.;") 
        single_doc = re.sub(pattern, "", single_doc) 
        # TR: Dokümanı küçük harflere çevir
        # EN: Convert document to lowercase
        single_doc = single_doc.lower()
        single_doc = single_doc.strip()
        # TR: Dokümanı token'larına ayır
        # EN: Tokenize documents
        tokens = WPT.tokenize(single_doc)
        # TR: Stop-word listesindeki kelimeler hariç al
        # EN: Filter out the stop-words 
        filtered_tokens = [token for token in tokens if token not in stop_words]
        # TR: Dokümanı tekrar oluştur
        # EN: Reconstruct the document
        single_doc = ' '.join(filtered_tokens)
        return single_doc

    def norm_values(perc_positive, perc_negative):
        sum = perc_positive + perc_negative
        norm_positive = perc_positive / sum
        norm_negative = perc_negative / sum
        
        return norm_positive, norm_negative

    def classify(message):
        message = re.sub('\W', ' ', message)
        message = norm_doc(message).split()

        p_positive_given_message = positive_percentage
        p_negative_given_message = negative_percentage

        for word in message:
            if word in parameters_positive:
                p_positive_given_message *= parameters_positive[word]
                
            if word in parameters_negative:
                p_negative_given_message *= parameters_negative[word]
        
        confidence = norm_values(p_positive_given_message, p_negative_given_message)

        if p_negative_given_message > p_positive_given_message:
            return True, confidence[0], confidence[1]
        elif p_negative_given_message < p_positive_given_message:
            return False, confidence[0], confidence[1]
    ########################################################################
    return classify(message)

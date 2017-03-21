import numpy as np
import string
import re
import os
import cPickle as pickle

class VocabularyIndex:

    """This class turns words into indices.

    It stores each UNIQUE word in a look up table.
    """

    def __init__(self, max_word_length=300):
        """Sets the max word length and creates an empty list."""
        self.max_length = max_word_length
        self.vocabulary_lookup_table = []

    def get_index(self, word):
        """Returns the index of the word.

        Arguments:
        word -- word whose index must be found or created.

        If the word doesn't exist, it appends the word to the list. Note this
        returns the index plus one because zero is used for padding.
        """
        if not word in self.vocabulary_lookup_table:
            self.vocabulary_lookup_table.append(word)
        return (self.vocabulary_lookup_table.index(word) + 1)

    def process(self, string):
        """This chops up a string into words and transforms the string into a
        list of indices for each word.
        """
        words = string.split()
        indices = []
        for i, word in enumerate(words):
            if i >= self.max_length:
                break
            indices.append(self.get_index(word))
        if len(indices) < self.max_length:
            indices += [0 for i in range(self.max_length - len(indices))]
        return indices, len(words)

    def vocab_size(self):
        """This returns the length of the lookup table of words."""
        return len(self.vocabulary_lookup_table)

def global_seed(seed):
    """Seed numpy."""
    np.random.seed(seed)

def load_csv(path):
    """Get a 2D numpy array of strings from a csv file.

    Arguments:
    path -- the path to the csv file.
    """
    return np.loadtxt(path, dtype='str', delimiter="\t", comments=None)

def strip_html_tags(html_string):
    """Strip HTML tags from string using regex.

    Arguments:
    html_string -- a string with html tags to be removed from it.

    Referenced from:
    http://stackoverflow.com/a/12982689/5398272
    """
    detagger = re.compile('<.*?>')
    return re.sub(detagger, '', html_string)

def strip_punctuation(input_string):
    """Strips all punctuation from input string.

    Arguments:
    input_string -- a string with punctuation to be removed from it.

    Referenced from:
    http://stackoverflow.com/a/266162/5398272
    """
    exclude = set(string.punctuation)
    return ''.join(char for char in input_string if char not in exclude)

def lower_case(input_string):
    """Removes any capital letters from string."""
    return input_string.lower()

def preprocess_words(input_string, vocab_lookup):
    """Removes html tags, punctuation and lower cases input string."""
    input_string = strip_html_tags(input_string)
    input_string = strip_punctuation(input_string)
    input_string = lower_case(input_string)
    indices, sequence_length = vocab_lookup.process(input_string)
    return indices, sequence_length

def files_to_examples():
    """Returns list of example dictionaries.

    This method will load each csv and and turn each row into a dictionary of
    three values. Here are the key value pairs:
        'rating' -- int with range 1 - 5
        'title'  -- string
        'review' -- string

    The function concatenates all of the rows from each csv into a single list
    from the dictionary. Each review for each example has it's html tags
    stripped as well as other preprocessing steps.

    Returns:
    all_examples -- list of dictionaries containing x and y data for supervised
                    learning.
    """
    vocab_lookup = VocabularyIndex()
    csv_paths = [
        'data/Andy-Weir-The-Martian.csv',
        'data/Donna-Tartt-The-Goldfinch.csv',
        'data/EL-James-Fifty-Shades-of-Grey.csv',
        'data/Fillian_Flynn-Gone_Girl.csv',
        'data/John-Green-The-Fault-in-our-Stars.csv',
        'data/Laura-Hillenbrand-Unbroken.csv',
        'data/Paula_Hawkins-The-Girl-On-The-Train.csv',
        'data/Suzanne-Collins-The-Hunger-Games.csv'
    ]
    stats = {
        'max_sequence_length' : 0,
        'vocabulary_size' : 0
    }
    examples_by_rating = [[] for i in range(5)]
    one_hot = np.eye(5)
    for path in csv_paths:
        csv_array = load_csv(path)
        for row in csv_array:
            review, length = preprocess_words(row[3], vocab_lookup)
            rating = int(float(row[0])) - 1
            example = {
                'rating'          : one_hot[rating],
                'review'          : review,
                'title'           : row[2],
                'sequence_length' : length
            }
            assert (rating < 5 and rating > -1)

            if len(example['review']) > stats['max_sequence_length']:
                stats['max_sequence_length'] = len(example['review'])
            examples_by_rating[rating].append(example)
    stats['vocabulary_size'] = vocab_lookup.vocab_size()

    examples_by_rating = sorted(examples_by_rating, key=lambda x: len(x))
    for i in range(5):
        print "rating: " + str(i + 1) + ", length: " + str(len(examples_by_rating[i]))

    smallest_rating_size = len(examples_by_rating[0])
    all_examples = []
    for i in range(5):
        all_examples += examples_by_rating[i][0:smallest_rating_size]

    print len(all_examples)

    return all_examples, stats

def train_test_valid_split(examples, train_amount, test_amount, valid_amount):
    """This splits the examples into test, train and validation sets."""
    assert (train_amount + test_amount + valid_amount) == 1.0
    assert examples != None and len(examples) != 0
    np.random.shuffle(examples)
    train = int(len(examples) * train_amount)
    test = train + int(len(examples) * test_amount)
    return examples[:train], examples[train:test], examples[test:]

def get_split_dataset(train_amount, test_amount, valid_amount):
    """"""
    files_exist = (os.path.isfile("train.p") and os.path.isfile("test.p") and
                   os.path.isfile("valid.p") and os.path.isfile("stats.p"))
    if files_exist:
        print "Loading Pickle files."
        train = pickle.load(open("train.p", "rb"))
        test = pickle.load(open("test.p", "rb"))
        valid = pickle.load(open("valid.p", "rb"))
        stats = pickle.load(open("stats.p", "rb"))
    else:
        print "Creating dataset and saving Pickle files."
        examples, stats = files_to_examples()
        train, test, valid = train_test_valid_split(examples, train_amount,
                                                    test_amount, valid_amount)
        pickle.dump(train, open("train.p", "wb"))
        pickle.dump(test, open("test.p", "wb"))
        pickle.dump(valid, open("valid.p", "wb"))
        pickle.dump(stats, open("stats.p", "wb"))
    return train, test, valid, stats

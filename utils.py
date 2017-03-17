import numpy as np
import re

def load_csv(path):
    """Get a 2D numpy array of strings from a csv file.

    Arguments:
    path -- the path to the csv file.
    """
    print np.loadtxt(path, dtype='str', delimiter=" ")
    return

def strip_html_tags(html_string):
    """Strip HTML tags from string using regex.

    Arguments:
    html_string -- a string with html tags to be removed from it.

    Referenced from:
    http://stackoverflow.com/a/12982689/5398272
    """
    detagger = re.compile('<.*?>')
    return re.sub(detagger, '', html_string)

def files_to_examples():
    """Returns list of example dictionaries.

    This method will load each csv and and turn each row into a dictionary of
    three values. Here are the key value pairs:
        'rating' -- int with range 1 - 5
        'title'  -- string
        'review' -- string

    The function concatenates all of the rows from each csv into a single list
    from the dictionary. Each review for each example has it's html tags
    stripped too.

    Returns:
    all_examples -- list of dictionaries containing x and y data for supervised
                    learning.
    """
    all_examples = []
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
    for path in csv_paths:
        csv_array = load_csv(path)
        for row in range(len(csv_array)):
            example = {
                'rating' : int(row[0]),
                'title'  : row[2],
                'review' : strip_html_tags(row[3])
            }
            all_examples.append(example)
    return all_examples

_ = files_to_examples()

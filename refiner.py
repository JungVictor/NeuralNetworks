#!/usr/bin/env python
# -*- coding: utf-8 -*-

import string

def string_hash(array):
    """
        Associate each element with an integer value.
        Return the hashed array and the hash table.

        :Example:
        >>> string_hash(["fr", "fr", "de", "it", "cn", "cn", "fr"])
        ([0, 0, 1, 2, 3, 3, 0], {'fr': 0, 'de': 1, 'it': 2, 'cn': 3})
    """
    counter = 0
    countries = {}
    countries_counter = {}
    column = []
    for element in array:
        if element in countries:
            column.append(countries[element])
            countries_counter[element] += 1
        else:
            column.append(counter)
            countries_counter[element] = 1
            countries[element] = counter
            counter += 1

    return column, countries


def keys_value(keys):
    """
        Associate to each key a power-2 value

        :Exemple:
        >>> keys_value(["fruit", "ripe", "smooth", "rustic", "herb"])
        ({'fruit': 1, 'ripe': 2, 'smooth': 4, 'rustic': 8, 'herb': 16}, {1: 'fruit', 2: 'ripe', 4: 'smooth', 8: 'rustic', 16: 'herb'})
    """
    value = 1
    values = {}
    reverse_values = {}
    for key in keys:
        values[key] = value
        reverse_values[value] = key
        value += value
    return values, reverse_values


def description_hash(array, keys):
    """
        Associate to each element of the array an integer value based on the keys.
        Return the hashed array and the values for each key.

        :Exemple:
        >>> description_hash(["This is fruity and smooth", "This is rustic and ripe"], ["fruit", "ripe", "smooth", "rustic", "herb"])
        ([5, 10], {'fruit': 1, 'ripe': 2, 'smooth': 4, 'rustic': 8, 'herb': 16}, {1: 'fruit', 2: 'ripe', 4: 'smooth', 8: 'rustic', 16: 'herb'})

        >>> description_hash(["This is fruity and smooth, with a rustic taste", "This is rustic and contains herbs"], ["fruit", "ripe", "smooth", "rustic", "herb"])
        ([13, 24], {'fruit': 1, 'ripe': 2, 'smooth': 4, 'rustic': 8, 'herb': 16}, {1: 'fruit', 2: 'ripe', 4: 'smooth', 8: 'rustic', 16: 'herb'})
    """
    values, reverse_values = keys_value(keys)
    column = []
    for element in array:
        value = 0
        for key in keys:
            if key in element:
                value += values[key]
        column.append(value)
    return column, values, reverse_values


def from_price_to_range(array, steps):
    """
        Associate each price to a price range, according to the steps array.

        :Exemple:
        >>> from_price_to_range([1,10,12,50,6,80], [10, 20, 40, 60, 80, 100])
        [0, 1, 1, 3, 0, 5]
    """
    range_number = {}
    steps.append(1000000)  # append infinity
    column = []
    for price in array:
        i = 0
        for price_range in steps:
            if price < price_range:
                column.append(i)
                if i in range_number:
                    range_number[i] += 1
                else:
                    range_number[i] = 0
                break
            else:
                i += 1
    return column, range_number

def count_words(array):
    """
        Count occurrences of a word in an array and return the most used ones
    """
    words = {}
    remove_punct_map = dict.fromkeys(map(ord, string.punctuation))
    useless = ["and", "of", "the", "a", "with", "is", "this", "in", "to", "that", "it", "from", "but", "are", "has", "for", "by", "drink", "it's", "finish", "as", "its", "an", "shows", "while", "at", "wine", "flavors", "on", "palate", "aromas", "acidity", "notes", "nose", "rich", "fresh", "now", "blend", "offers", "texture", "through", "well", "more", "full", "very", "good", "touch",  "some", "character", "will", "years", "out", "structure", "up", "or", "be", "not"]
    for description in array:
        for word in description.split():
            word = word.translate(remove_punct_map)
            if word.lower() not in useless:
                if word in words:
                    words[word] += 1
                else:
                    words[word] = 0
    top_words = sorted(words.items(), key=lambda k: k[1], reverse=True)
    top_words = top_words[:50]
    top_words = [k for (k, v) in top_words]
    redundancy = ""
    for word in top_words:
        if word not in redundancy:
            if word+"s" not in redundancy and word[:-1] not in redundancy:
                redundancy += word + " "
    return redundancy.split(sep=" ")


def from_int_to_list(n, table):
    """
        Convert an int to a list of element (reverse function of the description hashing)
    """
    res = []
    sorted_values = sorted(table.items(), key=lambda k: k[0], reverse=True)
    sorted_values = [k for (k, v) in sorted_values]
    for value in sorted_values:
        if n >= value:
            n -= value
            res.append(table[value])
    return res

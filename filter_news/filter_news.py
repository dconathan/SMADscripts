import json
import sys


def main(source):

    with open(source) as f:
        data = json.load(f)

    print("Loaded {} articles from {}".format(len(data), source))

    data = filter_by_score(filter_by_section(data), .6)

    print("Writing {} articles to {}".format(len(data), source + '.filtered'))

    with open(source + '.filtered', 'w') as f:
        json.dump(data, f)


def filter_by_section(data):
    """
    Pretty self-explanatory.
    Filters out any articles that match any sections in the "filter_out_sections.txt" file.
    """

    if not data:
        print("Input to filter_by_section is empty.")
        return data

    if 'section' not in data[0]:
        print("No 'section' field found in data. Bypassing filter_by_section filter.")
        return data

    with open('filter_out_sections.txt') as f:
        filter_out_sections = [s.strip() for s in f.readlines()]
    out = []
    filtered_out = 0
    assert 'section' in data[0]

    for d in data:
        include = True
        for s in filter_out_sections:
            if s.lower() in d['section'].lower():
                include = False
                break
        if include:
            out.append(d)
        else:
            filtered_out += 1

    print("Filtering by section filtered out {} articles, {} remain".format(filtered_out, len(out)))
    return out


def filter_by_score(data, threshold):
    """
    Again pretty self-explanatory.
    Filters out any articles that have a score lower than threshold.
    Must run train.py and score.py (in the scoring directory) first.
    """

    if not data:
        print("Input to filter_by_score is empty.")
        return data

    if 'score' not in data[0]:
        print("Data must be scored to run filter_by_score - bypassing this filter.")
        print("Look into the 'scoring' directory for more info")
        return data

    out = []
    filtered_out = 0

    for d in data:
        if d['score'] > threshold:
            out.append(d)
        else:
            filtered_out += 1

    print("Filtering by score filtered out {} articles, {} remain".format(filtered_out, len(out)))
    return out


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("\nUsage: \n\tpython filter_news.py INFILE\n\n\t"
              "INFILE:\t\tsource file from which to read and filter\n\nExample:"
              "\n\tpython filter_news.py examples/sample.json\n")
    else:
        main(sys.argv[1])

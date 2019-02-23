import pprint
import numpy as np
import calendar
from datetime import datetime
import json
import sys
import pickle
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)

def average_episode_length(data):
    ''' return the episode length '''
    try:
        runtime = data['imdb']['runtimes'][0]
    except:
        runtime = None
    logging.debug('runtime = {}'.format(runtime))
    return runtime

def years_running(data):
    ''' returns the number of years running '''
    return number_of_seasons(data)

def number_of_seasons(data):
    ''' returns the number of seasons '''
    try:
        seasons = len(data['imdb']['seasons'])
    except:
        seasons = None
    logging.debug('seasons = {}'.format(seasons))
    return seasons

def number_of_episodes(data):
    ''' returns the number of episodes '''
    try:
        episodes = data['imdb']['number of episodes']
    except:
        episodes = None
    logging.debug('episodes = {}'.format(episodes))
    return episodes

def genres(data):
    ''' returns a list of genres '''
    try:
        _genres = data['imdb']['genres']
    except:
        _genres = None
    logging.debug('genres = {}'.format(_genres))
    return _genres

def countries(data):
    ''' returns a list of countires '''
    try:
        _countries = data['imdb']['countries']
    except:
        _countries = None
    logging.debug('countries = {}'.format( _countries))
    return _countries

def languages(data):
    ''' returns a list of languages '''
    try:
        _languages = data['imdb']['languages']
    except:
        _languages = None
    logging.debug('languages = {}'.format(_languages))
    return _languages

def release_date(data):
    ''' returns the earliest release data in posix time format '''
    date_formats = ['%d %B %Y', '%B %Y', '%Y']

    try:
        dates = data['imdb']['raw release dates']
        date_objs = list()

        # read dates
        for rel_date in dates:
            for fmt in date_formats:
                try:
                    date_objs.append(datetime.strptime(rel_date['date'].strip(), fmt))
                except:
                    continue

        if len(date_objs) == 0:
            logging.debug('Invalide date format')
            logging.debug(dates)
            exit()

        earliest_date = min(date_objs)

        # convert date to posix time:
        _release_date = calendar.timegm(earliest_date.timetuple())

    except:
        _release_date = None
    logging.debug('release date = {}'.format(_release_date))
    return _release_date

def aspect_ratio(data):
    ''' return the series aspect ratio '''
    try:
        ratio = data['imdb']['aspect ratio']
    except:
        ratio = None
    logging.debug('aspect ratio = {}'.format(ratio))
    if ratio:
        return [ratio]
    return ratio 

def sound_mix(data):
    try:
        sound = data['imdb']['sound mix']
    except:
        sound = None
    logging.debug('sound mix = {}'.format(sound))
    return sound

def certificates(data):
    try:
        cert = data['imdb']['certificates']
    except:
        cert = None
    logging.debug('certificates = {}'.format(cert)) 
    return cert

def demographics(data) -> dict:
    ''' returns a dictionary with the demographics data '''
    try:
        demo = data['imdb']['demographics']
    except:
        demo = None
    logging.debug('demographics = {}'.format(demo))
    return demo

def number_of_shooting_locations(data):
    ''' returns a list of locations '''
    try:
        locations = data['imdb']['locations']
    except:
        locations = None
    logging.debug('locations = {}'.format(locations))
    return locations

def number_of_words_in_title(data):
    try:
        title = data['imdb']['title']
    except:
        title = []
    logging.debug('len title = {}'.format(len(title)))
    return len(title)

def number_of_writers(data):
    try:
        writers = data['imdb']['writer']
    except:
        writers = []
    logging.debug('num of writers = {}'.format(len(writers)))
    if writers:
        return len(writers)
    return -1

def distributors(data):
    try:
        dist = data['imdb']['distributors']
    except:
        dist = []
    logging.debug('distributors = {}'.format(len(dist)))
    return len(dist)

def rating_mean(data):
    try:
        mean = data['imdb']['arithmetic mean']
    except:
        mean = None

    logging.debug('rating mean = {}'.format(mean))
    return mean

def rating_median(data):
    try: 
        median = data['imdb']['median']
    except:
        median = None
    logging.debug('rating median = {}'.format(median))
    return median

def year(data):
    try:
        _year = data['imdb']['year']
    except:
        _year = None
    logging.debug('year = {}'.format(_year))
    return _year

extract_functions = {'runtime': average_episode_length,
             #'years_running': years_running,
             'seasons': number_of_seasons,
             'episodes': number_of_episodes,
             'release_date': release_date,
             'len_title': number_of_words_in_title,
             'num_writers': number_of_writers,
             'rating_mean': rating_mean,
             'rating_median': rating_median,
             'distributors': distributors,
             'year': year,
             'genres': genres,
             'countries': countries,
             'languages': languages,
             'aspect_ratio': aspect_ratio,
             'sound_mix': sound_mix,
             'certificates': certificates,
             'demographics': demographics,
             'locations': number_of_shooting_locations}

def get_all(key, data) -> set:
    ''' return a set of all the different values that appeared for the given
        key in the crawled data '''
    all_vals = set()

    # demographics special case
    if key == 'demographics':
        voter_data = ['votes', 'rating']
        for series in data:
            if series[key] is None:
                continue
            for voter_type in series[key]:
                for data_type in voter_data:
                    all_vals.add('{} {}'.format(voter_type, data_type))
    else:
        # other data types
        for series in data:
            if series[key] is None:
                continue
            for feature in series[key]:
                all_vals.add(clean_feature(feature, key))
                    
    return all_vals

def clean_feature(feature: str, key) -> str:
    '''
    cleans the given feature string from irrelevant notes
    'Alexandria, Minnesota, USA::(Baseball Park)' -> Alexandria, Minnesota, USA
    '''
    note_idx = feature.find('::')
    if note_idx > 0:
        clean = feature[:note_idx]
    else:
        clean = feature
    note_idx = clean.find('(')
    if note_idx > 0:
        clean = clean[:note_idx]

    if key == 'aspect_ratio':
        # special case: 'aspect_ratio 1.78 : 1 / '
        if clean[-2] == '/':
            clean = clean[:-3]

        # special case: '16 : 9' -> '16:9'
        if ' : ' in clean:
            clean = clean.replace(' : ', ':')

        # special case: '4: 3' -> '4:3'
        if '4: 3' in clean:
            clean = clean.replace('4: 3', '4:3')
    elif key == 'locations':
        # special case: reduce the location to city, country
        if clean.count(',') > 1:
            parts = [x.strip() for x in clean.split(',')]
            clean = ', '.join(parts[-2:])

    return clean.strip()


def vect_id(key, data, matrix, feature_index_map):
    for row, series in enumerate(data):
        col = feature_index_map[key]
        matrix[row, col] = series[key]


def vect_value(key, data, matrix, feature_index_map):
    feature_key = 'general {}'.format(key)

    # set the correct value
    for row, series in enumerate(data):
        col = feature_index_map[feature_key]
        val = series[key]
        if val is not None:
            matrix[row, col] = series[key]

def vect_list(key, data, matrix, feature_index_map):
    all_feature_keys = list(filter(lambda x: key in x, feature_index_map))
    feature_key_format = '{} {}'

    # set the correct values
    for row, series in enumerate(data):
        if series[key] is not None:
            for val in series[key]:
                feature_key = feature_key_format.format(key, clean_feature(val, key))
                col = feature_index_map[feature_key]
                matrix[row, col] = 1
            for feature_key in all_feature_keys:
                col = feature_index_map[feature_key]
                if matrix[row, col] == -1:
                    matrix[row, col] = 0
                    

def vect_demographics(key, data, matrix, feature_index_map):
    all_feature_keys = list(filter(lambda x: key in x, feature_index_map))
    data_points = ['votes', 'rating']
    key_format = key + ' {} {}'

    # set the data
    for row, series in enumerate(data):
        if series[key] is not None:
            for name, element in series[key].items():
                for data_type, val in element.items():
                    feature_key = key_format.format(name, data_type)
                    col = feature_index_map[feature_key]
                    matrix[row, col] = val

            for feature_key in all_feature_keys:
                col = feature_index_map[feature_key]
                if matrix[row, col] == -1:
                    matrix[row, col] == 0


vectorize_functions = {'id': vect_id,
             'runtime': vect_value,
             #'years_running': years_running,
             'seasons': vect_value,
             'episodes': vect_value,
             'release_date': vect_value,
             'len_title': vect_value,
             'num_writers': vect_value,
             'rating_mean': vect_value,
             'rating_median': vect_value,
             'distributors': vect_value,
             'year': vect_value,
             'genres': vect_list,
             'countries': vect_list,
             'languages': vect_list,
             'aspect_ratio': vect_list,
             'sound_mix': vect_list,
             'certificates': vect_list,
             'demographics': vect_demographics,
             'locations': vect_list}

def vectorize(data: list):
    # initialize vectorized data structure
    feature_index_map = get_column_map(data)
    matrix = init_matrix(data, feature_index_map)
    logging.info('vectorizing data...')

    for key, vect_func in vectorize_functions.items():
        logging.info('vectorizing {}...'.format(key))
        sys.stdout.flush()
        vect_func(key, data, matrix, feature_index_map)
    return matrix

def extract_data(paths) -> list:
    data = list()
    for path in paths:
        for json_file_path in Path(path).iterdir():
            series = dict()
            series_id = int(str(json_file_path).split('/')[-1].split('.')[0][2:])
            series['id'] = series_id

            logging.info('extracting data from {}'.format(json_file_path))
            with open(str(json_file_path), 'r') as f:
                json_file = json.load(f, encoding='utf8')

            for key, func in extract_functions.items():
                series[key] = func(json_file)
            data.append(series)
    return data

def get_column_map(data):
    ''' returns a feature -> matrix column map '''
    voter_data = ['votes', 'rating']
    index_feature_map = dict()
    index_feature_map['id'] = 0
    i = 1
    for key, func in vectorize_functions.items():
        if func is vect_value:
            index_feature_map['general {}'.format(key)] = i
            i += 1
        if func is vect_list:
            all_options = get_all(key, data)
            for option in all_options:
                index_feature_map['{} {}'.format(key, option)] = i
                i += 1
        if func is vect_demographics:
            voter_types = get_all(key, data)
            for voter_type in voter_types:
                index_feature_map['{} {}'.format(key, voter_type)] = i
                i += 1
    return index_feature_map

def init_matrix(data, column_map):
    return np.full((len(data), len(column_map)), -1, np.float)

if __name__ == '__main__':
    _data = extract_data(sys.argv[1:])
    _feature_map = get_column_map(_data)
    with open('pickled_data.pickle', 'wb') as _f:
        _result = vectorize(_data)
        for _key, _val in _feature_map.items():
            print('{} {}'.format(_key, _result[1, _val]))
        pickle.dump((_feature_map, _result), _f)

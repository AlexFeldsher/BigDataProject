import wikipedia
import wikipediaapi  # backup api
import json
import argparse
import os
import time
import csv
from imdb import IMDb

# IMDB available data: print(ia.get_movie_infoset())
'''
ALL_INFO = ['airing', 'akas', 'alternate versions', 'awards', 'news',
             'connections', 'crazy credits', 'critic reviews', 'episodes',
             'external reviews', 'external sites', 'faqs', 'full credits',
             'goofs', 'keywords', 'locations', 'main', 'misc sites',
             'official sites', 'parents guide', 'photo sites', 'plot',
             'quotes', 'release dates', 'release info', 'reviews',
             'sound clips', 'soundtrack', 'synopsis', 'taglines', 'technical',
             'trivia', 'tv schedule', 'video clips', 'vote details']
'''

IMDB_INFO = ['airing', 'akas', 'alternate versions', 'awards', 'connections',
             'critic reviews', 'episodes', 'faqs', 'full credits', 'keywords',
             'locations', 'main', 'parents guide', 'plot', 'release dates',
             'release info', 'synopsis', 'technical', 'tv schedule', 'vote details']

ia = IMDb()

# backup wiki library
wiki_backup_html = wikipediaapi.Wikipedia(language='en',
                                          extract_format=wikipediaapi.ExtractFormat.WIKI)
wiki_backup_text = wikipediaapi.Wikipedia('en')

# imdb database series type names
SERIES_TYPES = ['tvMiniSeries', 'tvSeries']
# imdb database csv file index
ID = 0          # series IMDB id
TYPE = 1
TITLE = 2
START_YEAR = 5


def crawl(imdb_title_db, output_dir, year):
    db = open(imdb_title_db, 'r')
    reader = csv.reader(db, delimiter='\t')
    for i, line in enumerate(reader):
        series_id = line[ID]
        series_name = line[TITLE]

        # some series have no start year in the db
        try:
            series_start_year = int(line[START_YEAR])
        except Exception:
            # no data on series start year
            series_start_year = 0

        # skip irelevant movies and previously crawled series
        crawled = os.path.isfile('{}/{}.json'.format(output_dir, series_id))
        if line[TYPE] not in SERIES_TYPES or crawled or year != series_start_year:
            continue

        print('#{} :: id: {}, name: {}'.format(i + 1, series_id, series_name))

        data = dict()
        data['imdb'] = scrape_imdb_page(series_id[2:], IMDB_INFO)
        if data['imdb'] is not None:
            data = scrape_wikipedia_page(data, data['imdb']['title'], series_start_year)
            with open('{}/{}.json'.format(output_dir, series_id), 'w') as out:
                out.write(json.dumps(data, ensure_ascii=False, default=lambda o: o.data, indent=4))
    db.close()


def imdb_to_json(data):
    json_str = json.dumps(data, ensure_ascii=False, default=lambda o: o.data, indent=4)
    return json.loads(json_str)


def scrape_wikipedia_page(data: dict, title, year) -> dict:
    """ Search for '<title> (<year> TV series)' and scrape the page """
    search = wikipedia.search('{} ({} TV series)'.format(title, year))
    data['wiki'] = None

    if len(search) == 0:
        print("Didn't find wikipedia page...")
        return data

    # Try to scrape a relevant page
    # wiki title and series title must have 1 word incommon
    title_words = set(title.split(' '))
    best_result_words = set(search[0].split(' '))
    if len(title_words.intersection(best_result_words)) == 0:
        print("Didn't find wikipedia page...")
        return data

    data['wiki'] = dict()
    data['wiki']['title'] = search[0]
    print('Scraping Wikipedia page "{}"...'.format(search[0]))
    try:
        page = wikipedia.page(search[0])
        data['wiki']['url'] = page.url
        data['wiki']['categories'] = page.categories
        data['wiki']['content'] = page.content
        data['wiki']['html'] = page.html()
        data['wiki']['links'] = page.links
        data['wiki']['summary'] = page.summary
    except Exception:
        print('Using backup wikipedia scraper')
        page_html = wiki_backup_html.page(search[0])
        page_text = wiki_backup_text.page(search[0])
        data['wiki']['url'] = page_html.fullurl
        data['wiki']['categories'] = [x[x.find(':') + 1:] for x in page_html.categories.keys()]
        data['wiki']['content'] = page_text.text
        data['wiki']['html'] = page_html.text
        data['wiki']['links'] = list(page_html.links.keys())
        data['wiki']['summary'] = page_text.summary

    return data


def scrape_imdb_page(series_id, info):
    """ :param series_id: id with the prefix 'tt' removed
        :para info: the info the gather
        :returns: the series data object, of None if failed """
    print('Scraping IMDB...')
    data = ia.get_movie(series_id, info)
    return imdb_to_json(data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='IMDB crawler')
    parser.add_argument('title_tsv', type=str, nargs=1, help='imdb title database path')
    parser.add_argument('year', type=int, nargs=1, help='series year to crawl')
    parser.add_argument('output', type=str, nargs=1, help='Output directory')
    args = parser.parse_args()

    if not os.path.isfile(args.title_tsv[0]):
        print('wrong title.basics.tsv path')
        exit()
    if not os.path.isdir(args.output[0]):
        print("output directory doesn't exist")
        exit()

    crawl(args.title_tsv[0], args.output[0], args.year[0])

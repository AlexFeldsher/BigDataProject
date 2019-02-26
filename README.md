# BigDataProject

### Crawler
```
usage: crawler.py [-h] title_tsv year output

IMDB crawler

positional arguments:
  title_tsv   imdb title database path
  year        series year to crawl
  output      Output directory

optional arguments:
  -h, --help  show this help message and exit
```
### Convert crawled data to pandas DataFrame
```
usage: vectorize_data.py [-h] -i INPUT [INPUT ...] -o OUTPUT

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT [INPUT ...], --input INPUT [INPUT ...]
                        <Required> input directory
  -o OUTPUT, --output OUTPUT
                        <Required> output file path
 ```

from gutenberg import acquire
from gutenberg import cleanup
import re

def generate_samples(data_dir, tmp_dir, dataset_split):
    del data_dir
    del tmp_dir
    del dataset_split


    books = [
        # bookid, skip N lines
        (19221, 223),
        (15553, 522),
    ]

    for (book_id, toskip) in books:
      text = cleanup.strip_headers(acquire.load_etext(book_id)).strip()
      lines = text.split("\n")[toskip:]
      prev_line = None
      ex_count = 0
      for line in lines:
        # Any line that is all upper case is a title or author name
        if not line or line.upper() == line:
          prev_line = None
          continue

        line = re.sub("[^a-z]+", " ", line.strip().lower())
        
        if prev_line and line:
          yield {
              "inputs": prev_line,
              "targets": line,
          }
          ex_count += 1
        prev_line = line

a,b,c=[],[],[]
for i in generate_samples(a,b,c):
    print(i)
    break
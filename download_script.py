import hashlib
import requests
import csv
import random as r
import shutil
import os
import time

def makeUrl(full_name, wiki):
    name, ext = os.path.splitext(full_name)
    if ext == ".svg":
        ext = ".svg.png"
    else:
        ext = ".png"
    size = 50
    h = hashlib.md5(full_name.encode('utf-8')).hexdigest()
    url = "https://upload.wikimedia.org/wikipedia/" + wiki + "/thumb/" + h[:1] + "/" + h[:2] + "/" + full_name + "/" + str(size) + "px-" + name + ext
    return url

def download(file_list, output_directory, wiki):
    with open(file_list) as f:
        reader = csv.reader(f)
        chosen_rows = r.sample(list(reader),105)
        for l in chosen_rows:
            full_name_no_File = l[1][5:]
            url = makeUrl(full_name_no_File, wiki)
            response = requests.get(url, stream=True)

            name, ext = os.path.splitext(full_name_no_File)
            save_as_name = name + ".png"

            with open(output_directory + '/' + save_as_name, 'wb') as out_file:
                shutil.copyfileobj(response.raw, out_file)
            del response
            time.sleep(1)

def main():
    # download from Wikipedia
    download("./Wikipedia_Non-free_logos.csv", "./above_data_eval/", "en")
    # download from Commons
    download("./Commons_PD_text_logos.csv", "./below_data_eval/", "commons")

if __name__ == "__main__":
    main()

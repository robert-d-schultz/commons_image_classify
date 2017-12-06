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
    if ext == ".tiff":
        ext = ".tiff.png"
    if ext == ".tif":
        ext = ".tif.png"
    size = 50
    h = hashlib.md5(full_name.encode('utf-8')).hexdigest()
    url = "https://upload.wikimedia.org/wikipedia/" + wiki + "/thumb/" + h[:1] + "/" + h[:2] + "/" + full_name + "/" + str(size) + "px-" + name + ext
    return url

def download(file_list, output_directory, wiki):
    with open(file_list) as f:
        reader = csv.reader(f)
        chosen_rows = r.sample(list(reader),20)
        #chosen_rows = list(reader)
        #chosen_rows = chosen_rows[1:]
        for l in chosen_rows:
            full_name_no_File = l[1][5:]
            url = makeUrl(full_name_no_File, wiki)
            print(url)
            response = requests.get(url, stream=True)

            name, ext = os.path.splitext(full_name_no_File)
            if ext == ".svg":
                ext = ".svg.png"
            if ext == ".tiff":
                ext = ".tiff.png"
            if ext == ".tif":
                ext = ".tif.png"
            save_as_name = name + ext

            with open(output_directory + '/' + save_as_name, 'wb') as out_file:
                shutil.copyfileobj(response.raw, out_file)
            del response
            time.sleep(1)


def main():
    # download from Wikipedia
    download("./Wikipedia_Non-free_logos.csv", "./above_data_eval/", "en")
    #download("./Wikipedia and below.csv", "./below_data_eval2/", "en")
    # download from Commons
    download("./Commons_PD_text_logos.csv", "./below_data_eval/", "commons")
    #download("./Commons and below.csv", "./below_data_eval2/", "commons")



    #with open("./Commons_PD_text_logos.csv") as f:
    #    reader = csv.reader(f)
    #    chosen_rows = r.sample(list(reader),500)
    #    print(["https://commons.wikimedia.org/wiki/" + l[1] for l in chosen_rows])

if __name__ == "__main__":
    main()

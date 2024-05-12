import numpy as np
import os 

import os 
import requests

def download_data_from_url(url, file_path, file_name):
    file = os.path.join(file_path, file_name)

    if not os.path.exists(file_path):
        os.makedirs(file_path)

    if not os.path.exists(file):
        response = requests.get(url)
        if response.status_code == 200:
            with open(file, 'wb') as f:
                f.write(response.content)
            print("successfully download the data")
        else:
            print("failed to download from the url")
    else:
        print("data file already exist")


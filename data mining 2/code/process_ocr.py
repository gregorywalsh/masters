from pathlib import Path
import operator
from bs4 import BeautifulSoup

root = Path('/Users/gregwalsh/Google Drive/Study/Data Science Masters/Modules/Semester 2/Data Mining/CW2/Data/gap-html')

folder_paths = [f for f in root.iterdir() if f.is_dir()]

corpus = {}

for folder_path in folder_paths:

    plain_text = ''
    file_processed_count = 0
    html_file_paths = list(folder_path.glob('**/*.html'))
    html_file_paths.sort(key=operator.attrgetter("stem"))
    for html_file_path in html_file_paths:


        with open(html_file_path, 'r') as file:
            html_contents = file.read()

        soup = BeautifulSoup(html_contents, 'html.parser')
        ocrx_blocks = soup.find_all("div", class_="ocrx_block")
        for ocrx_block in ocrx_blocks:
            plain_text += ocrx_block.get_text()

        # TODO - Possible OCR cleanup

        file_processed_count += 1
        print(file_processed_count)

    with open('corpus/{0}.txt'.format(folder_path.name), 'w+') as f:
        f.write(plain_text)
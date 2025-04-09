import os
import glob
import random
import fitz
from tqdm import tqdm
import signal

INPUT_DIR = "./climate_text_dataset"
OUTPUT_DIR = "./processed_data"
TRAIN_DIR = os.path.join(OUTPUT_DIR, "train")
TEST_DIR = os.path.join(OUTPUT_DIR, "test")

os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(TEST_DIR, exist_ok=True)

class TimeoutException(Exception):
    pass

def handler(signum, frame):
    raise TimeoutException()

signal.signal(signal.SIGALRM, handler)

pdfs = []
for root, dirs, files in os.walk(INPUT_DIR):
	for file in files:
		if file.endswith('.pdf'):
			pdfs.append(os.path.join(root, file))

print(f"total number of PDFS = {len(pdfs)}")

random.seed(42)
random.shuffle(pdfs)
split_idx = int(len(pdfs) * 0.9)
train_files = pdfs[:split_idx]
test_files = pdfs[split_idx:]

print(f"train set size = {len(train_files)}")
print(f"test set size = {len(test_files)}")

def extract_text(file_path):
	text = ""
	try:
		doc = fitz.open(file_path)
		for page in doc:
			page_text = page.get_text()
			if page_text:
				text += page_text + "\n\n"
		doc.close()
	except Exception as e:
		print(f"error processing {file_path}: {e}")
	return text

print("processing train files")
for i, file_path in enumerate(tqdm(train_files)):
    try:
        signal.alarm(15)  # 15 second timeout per file
        text = extract_text(file_path)
        signal.alarm(0)  # disable alarm
        if text.strip():
            file_name = f"train_{i}.txt"
            with open(os.path.join(TRAIN_DIR, file_name), 'w', encoding='utf-8') as f:
                f.write(text)
    except TimeoutException:
        print(f"Timeout: Skipping {file_path}")
    except Exception as e:
        print(f"Unexpected error on {file_path}: {e}")

print("processing test files")
for i, file_path in enumerate(tqdm(test_files)):
    try:
        signal.alarm(15)
        text = extract_text(file_path)
        signal.alarm(0)
        if text.strip():
            file_name = f"test_{i}.txt"
            with open(os.path.join(TEST_DIR, file_name), 'w', encoding='utf-8') as f:
                f.write(text)
    except TimeoutException:
        print(f"Timeout: Skipping {file_path}")
    except Exception as e:
        print(f"Unexpected error on {file_path}: {e}")

print("text extraction complete")
print(f"train files saved to: {TRAIN_DIR}")
print(f"test files saved to: {TEST_DIR}")
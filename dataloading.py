# Data Loading Functions & Preprocessing Functions

# Pull data from the dataset training or testing folders and return a list of tuples containing the filename and the tokenized message content.


import os
import re
import csv
import json
from collections import Counter
import datetime
import math

#!. Initialization & Data Loading
local_drive_path = "/project_SPF170/dataset/training"  # Define the local drive path for output files

# Date,Rainfall(mm),2m Temp(C),Heat Index(C),Relative Humidity(%),Wind Speed(m/s),Wind Direction,Pressure(hPa),Weather
feature = ['date', 'rainfall', 'temp', 'heat_index', 'humidity', 'wind_speed', 'pressure']
feature_text = ['wind_direction', 'weather']
csv_to_field = {
    'Date': 'date', 'Rainfall(mm)': 'rainfall', '2m Temp(C)': 'temp',
    'Heat Index(C)': 'heat_index', 'Relative Humidity(%)': 'relhum',
    'Wind Speed(m/s)': 'wind_speed', 'Wind Direction': 'wind_dir',
    'Pressure(hPa)': 'pressure', 'Weather': 'weather',
}

def parse_filename(filename):           # Extract the municipality and province from the filename
    
    match = re.search(r'(\d{8})_(\d{6})', filename)
    if match:
        date_str = match.group(1)  # YYYYMMDD
        time_str = match.group(2)  # HHMMSS
        date_time_str = f"{date_str} {time_str}"
        date_time_obj = datetime.datetime.strptime(date_time_str, '%Y%m%d %H%M%S')
        return date_time_obj
    else:
        return None
    
def tokenize(directory_path):                                                 # tokenize the contents of a directory; returns list of (filename, tokens)
    tokenized_message_list = []

    if not os.path.exists(directory_path):
        print(f"Directory '{directory_path}' does not exist.")
        return tokenized_message_list

    for filename in os.listdir(directory_path):                               # outer loop, check each file inside the directory
        if filename.endswith(".eml"):                                          # checks the file that ends with ".eml" extension or email
            file_path = os.path.join(directory_path, filename)                # create a full path to the file.
            print(file_path)
            try:
                with open(file_path, 'r', encoding='Latin-1') as f:           # open the file, read in Latin-1 encoding and pass contents into a ContentArray
                    content = f.read()
                    cleaned_string = clean(content)                           # Clean the message string
                    tokenized_message = cleaned_string.split()                # Tokenize cleaned string
                    tokenized_message_list.append((filename, tokenized_message)) # Append (filename, tokenized_message) tuple
            except Exception as e:
                print(f"Error reading {filename}: {e}")

    return tokenized_message_list                                             # end TOKENIZE Function
    # TOKENIZE

def BagofWords(directory_path):                             # Generate a bag of words from cleaned text

  tokenized_data = tokenize(directory_path)                 # retrieve the tokenized tuples from the directory,
  dictionary = {}                                           # create a dictionary with word count
  for filename, tokens in tokenized_data:
    for word in tokens:
      if word not in dictionary:
        dictionary[word] = 1
      else:
        dictionary[word] += 1
  return dictionary

def main():
    directory_path = "/dataset/training/"               # Define dataset path for training
    bag_of_words = BagofWords(directory_path)
    print(bag_of_words)

    BoW = BagofWords(directory)

BoW_outputfile = os.path.join(local_drive_path, 'output.txt')

try:
  with open(BoW_outputfile, 'w', encoding='utf-8') as outfile:
    outfile.write(f"Dictionary Size:  {len(BoW)}\n")                   # print the dictionary size
    outfile.write(f"Total Word Count:  {sum(BoW.values())}\n\n")       # print the total word count
    outfile.write("---DICTIONARY VOCABULARY---\n")
    for word, count in sorted(BoW.items()):                            # sort the vocabulary alphabetically
      outfile.write(f"{word} : {count}\n")

except IOError as e:
  print(f"\n NOT SUCCESSFUL. Error writing to file {BoW_outputfile}: {e}")

  drive.flush_and_unmount()
print("Drive flushed and unmounted. Re-run the mount cell before using Drive again.")
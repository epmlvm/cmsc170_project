
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
output_file_path = os.path.join(local_drive_path, 'output.txt')  # Define the output file path

# Date,Rainfall(mm),2m Temp(C),Heat Index(C),Relative Humidity(%),Wind Speed(m/s),Wind Direction,Pressure(hPa),Weather
feature = ['date', 'rainfall', 'temp', 'heat_index', 'humidity', 'wind_speed', 'pressure']
feature_text = ['wind_direction', 'weather']
csv_to_field = {
    'Date': 'date', 'Rainfall(mm)': 'rainfall', '2m Temp(C)': 'temp',
    'Heat Index(C)': 'heat_index', 'Relative Humidity(%)': 'relhum',
    'Wind Speed(m/s)': 'wind_speed', 'Wind Direction': 'wind_dir',
    'Pressure(hPa)': 'pressure', 'Weather': 'weather',
}

output_fields = ['date', 'rainfall', 'temp', 'heat_index', 'relhum', 'wind_speed', 'wind_dir', 'pressure', 'weather']

def parse_filename(filename):           # Extract the municipality and province from the filename
  match = re.search(r'^MUN_(.+)_([^_]+)$', filename)
  if match:
    return match.group(1).replace('_',' '), match.group(2)
  return None, None

def parse_foldername(foldername):           # Extract the municipality and province from the filename
  match = re.search(r'_(\d{10})_', foldername)
  if match:
    return datetime.datetime.strptime(match.group(1), '%Y%m%d%H')
  return None

def to_number(value):  # Convert a string from the data to a number (int or float)
  try:
    return int(value)
  except ValueError:
    return None

def split_weather(code):
  match = re.match(r'(\d+)([A-Z]*)', code)
  if match:
    level = int(match.group(2)) if match.group(2) else None
    return match.group(1), level, match.group(3)
  return code, None, None

def read_weather_file(file_path):  # Read a weather data file and return a list of dictionaries
  rows = []
  with open(file_path,'r', encoding='utf-8') as csvfile:  # open the file, read in utf-8 encoding and pass contents into a ContentArray
    reader = csv.DictReader(f)
    for row in reader:
      row = {}
      for csv_col, value in record.items():
        if csv_col == None:
           continue
        value = (value or '').strip()
        if field == 'date':
          try:
             row['date'] = datetime.datetime.strptime(value, '%Y-%m-%d %H:%M:%S') # Tokenize cleaned string/row
          except ValueError:
             row['date'] = None
        elif field_name in feature:
           row[field_name] = to_number(value)
        else:
           row[field_name] = value
        row['weather_type'], row['weather_level'], row['day_night'] = split_weather(row.get('weather', ''))
        rows.append(row)
  return rows


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


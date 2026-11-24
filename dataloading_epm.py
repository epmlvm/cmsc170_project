
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
local_drive_path = "/project_SPF170/dataset/"  # Define the local drive path for output files
output_file_path = os.path.join(local_drive_path, 'output.txt')  # Define the output file path
frontier_path = os.path.join(local_drive_path, 'frontier.json')  # Frontier path

folder_names = {'1': 'training', '2': 'testing', '3': 'implement'} # let user choose what to do
default_municipality = ('LosBanos', 'Laguna')             # test for LosBanos, Laguna
num_samples = 10000                                       # n samples for testing
alpha = 1                                                 # for laplace
random.seed(170)                                          # reproducible sampling


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

def tokenize_weather_file(file_path):  # Read a weather data file and return a list of dictionaries # modification to prev tokenize function
  rows = []
  with open(file_path,'r', encoding='utf-8') as csvfile:  # open the file, read in utf-8 encoding and pass contents into a ContentArray
    reader = csv.DictReader(csvfile)
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


# function to load and tokenaize all the data from all the subfolders of the subfolders. each subfolder under the training folder would be one month. the folder name is the year and month.

def load_and_tokenize_data(directory_path):  # Load and tokenize all data from subfolders
  tokenized_data = []
  file_count = Counter()

  if not os.path.exists(dataset_path):
    print("Dataset path does not exist.")
  for root, dirs, files in os.walk(directory_path):
    for file in files:
      if file.endswith('.csv'):
        file_path = os.path.join(root, file)
        tokenized_data.extend(tokenize_weather_file(file_path))
  return tokenized_data

def WeatherForecast_BOW(dataset_path):                             # Generate a bag of words from cleaned text
  tokenized_weatherdata = tokenize_weather_file(dataset_path)    # retrieve the tokenized tuples from the dataset_path,
  weather_dictionary = {}                                           # create a dictionary with word count
  for filename, tokens in tokenized_weatherdata:
    for daily_weather in tokens:
      if daily_weather not in weather_dictionary:
        weather_dictionary[daily_weather] = 1
      else:
        weather_dictionary[daily_weather] += 1
  return weather_dictionary

def main():
    global local_drive_path

    while True:           # Define dataset path for training
      if folder_choice == '1':
        mode = os.path.join(local_drive_path, 'training/201705')
        break
      elif folder_choice == '2':
        mode = os.path.join(local_drive_path, 'testing/201804')
        break
      elif folder_choice == '3':
        mode = os.path.join(local_drive_path, 'implementation')
        break  
      else:
        if folder_choice not in ['1', '2', '3']:
          print("Invalid choice. Please enter '1', '2', or '3'.")
        folder_choice = input("Choose Folder : [1] training | [2] testing | [3] implementation : ")

    dataset_path = os.path.join(local_drive_path, mode)

    weather_data = load_and_tokenize_data(dataset_path)  # Load and tokenize data from the dataset path
    print(f"Loaded and tokenized {len(weather_data)} weather records from {dataset_path}")

    frontier = None

    if mode != 'training':
      frontier = load_frontier(frontier_path)
      if frontier is None:
        print(f"\n Frontier is Emplty, Run Training first.")
        return
      print(f"Frontier loaded."
            )

    tokenized_data, file_count = load_and_tokenize_data(dataset_path)
    if not tokenized_data:
      print(f"\n Not Successful. No data found under {dataset_path}")
      return

    weather_dictionary = weather_forecast_BoW(tokenized_data)
    additional_data = []
    feature = output_fields

    if mode =='training':
      frontier = build_frontier(tokenized_data, file_count)
      save_frontier(frontier, frontier_path)
      additional_data.append(f"Frontier Saved: {frontier_path} ({len(frontier['locations'])} locations)")
    

    try:
      if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")
      for folder in os.listdir(dataset_path):
        if not os.path.isdir(os.path.join(dataset_path, folder)):
          raise NotADirectoryError(f"Expected a directory but found a file: {folder}")
        #check the sub folders of this folder for csv or files, that start with MUN_
        #parse the filename, and store the data in a dictionary with the municipality and province as keys, and the tokenized data as values
        for subfolder in os.listdir(os.path.join(dataset_path, folder)):
          if not os.path.isdir(os.path.join(dataset_path, folder, subfolder)):
            raise NotADirectoryError(f"Expected a directory but found a file: {subfolder}")
          for file in os.listdir(os.path.join(dataset_path, folder, subfolder)):
            if not file.endswith('.csv'):
              raise ValueError(f"Expected a CSV file but found: {file}")
            if not file.startswith('MUN_'):
              raise ValueError(f"Expected a file starting with 'MUN_' but found: {file}")
        location_classification = tokenize_location_file(dataset_path)


BoW_outputfile = os.path.join(local_drive_path, 'output.txt')

try:
  with open(BoW_outputfile, 'w', encoding='utf-8') as outfile:
    outfile.write(f"Total Daily Weather Studied:  {len(BoW)}\n")                   # print the dictionary size
    outfile.write(f"Total Provinces:  {sum(BoW.values())}\n\n")       # print the total word count
    outfile.write(f"Total Municipalities:  {sum(BoW.values())}\n\n")       # print the total word count
    outfile.write(f"Total Daily Weather Records:  {sum(BoW.values())}\n\n")       # print the total word count
    outfile.write("---DICTIONARY VOCABULARY---\n")
    for word, count in sorted(BoW.items()):                            # sort the vocabulary alphabetically
      outfile.write(f"{word} : {count}\n")

except IOError as e:
  print(f"\n NOT SUCCESSFUL. Error writing to file {BoW_outputfile}: {e}")

main()
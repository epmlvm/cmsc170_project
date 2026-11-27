
import os
import re
import sys
import csv
import json
import math
import random
import datetime
from collections import Counter, defaultdict

#1. Initialization
script_folder = os.path.dirname(os.path.abspath(__file__))
local_drive_path = os.path.join(script_folder, 'dataset')
if len(sys.argv) > 1:                                         
  local_drive_path = sys.argv[1]
output_file_path = os.path.join(local_drive_path, 'output.txt')   # Define the output file path
frontier_path = os.path.join(local_drive_path, 'frontier.json')   # Frontier path (saved Bayes net CPTs)

folder_names = {'1': 'training', '2': 'testing', '3': 'implement'} # let user choose what to do
folder_months = {                                           # CHANGED: month folders read per mode ([] = the whole mode folder)
  'training': ['201511', '201704', '201705', '201706', '201707'],
  'testing': ['201804'],
  'implement': [],
}
outlook_prefix = 'weather-outlook'
use_outlook_folders = {'training': True, 'testing': True, 'implement': False}

test_municipality = {                                              
  '1': ('Manila', 'MetropolitanManila'),
  '2': ('LosBanos', 'Laguna'),
  '3': ('Balabac', 'Palawan'),
}

num_samples = 10000                                                 # n samples per Likelihood Weighting query
alpha = 1                                                           # for laplace
random.seed(170)                                                    # reproducible sampling

observed_days = 4                       
forecast_days = 4                       
slot_hours = 3                          
use_trend = False                       

# Date,Rainfall(mm),2m Temp(C),Heat Index(C),Relative Humidity(%),Wind Speed(m/s),Wind Direction,Pressure(hPa),Weather
feature = ['rainfall', 'temp', 'heat_index', 'relhum', 'wind_speed', 'pressure']   # numeric fields ('humidity' -> 'relhum')
feature_text = ['wind_dir', 'weather']                                              # text fields ('wind_direction' -> 'wind_dir')
csv_to_field = {
    'Date': 'date', 'Rainfall(mm)': 'rainfall', '2m Temp(C)': 'temp',
    'Heat Index(C)': 'heat_index', 'Relative Humidity(%)': 'relhum',
    'Wind Speed(m/s)': 'wind_speed', 'Wind Direction': 'wind_dir',
    'Pressure(hPa)': 'pressure', 'Weather': 'weather',
}
output_fields = ['date', 'rainfall', 'temp', 'heat_index', 'relhum', 'wind_speed', 'wind_dir', 'pressure', 'weather']

all_weather_classes = ['Clear', 'Cloud1', 'Cloud2', 'Rain0', 'Rain1', 'Rain2']        # NEW: query variable values
evidence_features = ['rain_rate', 'temp', 'relhum', 'wind_speed', 'pressure', 'wind_dir']  # NEW: evidence nodes (children of W)

feature_bins = {                                                    # NEW: (upper limit, label) used to discretize numbers
  'rain_rate':  [(0.0, 'none'), (2.5, 'light'), (7.5, 'moderate'), (math.inf, 'heavy')],   # mm/hr (PAGASA intensity)
  'temp':       [(26.0, 'cool'), (30.0, 'warm'), (math.inf, 'hot')],                       # C
  'relhum':     [(70.0, 'dry'), (85.0, 'humid'), (math.inf, 'very_humid')],                # %
  'wind_speed': [(2.0, 'calm'), (5.0, 'light'), (math.inf, 'breezy')],                     # m/s
  'pressure':   [(1009.0, 'low'), (1011.0, 'normal'), (math.inf, 'high')],                 # hPa
}
compass16 = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']
compass8 = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']             # NEW: wind_dir collapsed to 8 points (less sparse CPT)

#2. Parsing helpers

def parse_filename(filename):           # Extract the municipality and province from the filename
  base = os.path.splitext(os.path.basename(filename))[0]             # drop folder + .json/.csv extension
  match = re.search(r'MUN_(.+)_([^_]+)$', base)                      # no '^' so prefixes like '947ac8e7-MUN_...' still work
  if match:
    return match.group(1).replace('_', ' '), match.group(2)
  return None, None

def parse_foldername(foldername):       # Extract the model initialization datetime from a folder name (..._YYYYMMDDHH_...)
  match = re.search(r'_(\d{10})_', foldername)
  if match:
    return datetime.datetime.strptime(match.group(1), '%Y%m%d%H')
  return None

def to_number(value):  # Convert a value from the data to a float; ranges like "27.8-31.9" become the midpoint
  if value is None:
    return None
  if isinstance(value, (int, float)):
    return float(value)
  value = str(value).strip()
  match = re.fullmatch(r'(\d+(?:\.\d+)?)-(\d+(?:\.\d+)?)', value)
  if match:
    return (float(match.group(1)) + float(match.group(2))) / 2
  try:
    return float(value)
  except ValueError:
    return None

def split_weather(code):  # "Cloud1N" -> ('Cloud', 1, 'N') ; "ClearD" -> ('Clear', None, 'D')
  match = re.match(r'^([A-Za-z]+?)(\d*)([DN])$', (code or '').strip())
  if match:
    level = int(match.group(2)) if match.group(2) else None
    return match.group(1), level, match.group(3)
  return code, None, None

def weather_class(row):   # NEW: query value = type + level, e.g. 'Rain1' (day/night is its own node)
  if row.get('weather_type') is None:
    return None
  return row['weather_type'] + (str(row['weather_level']) if row['weather_level'] is not None else '')

def day_night(date):      # NEW: DN evidence from the clock
  return 'D' if 6 <= date.hour < 18 else 'N'

def to_compass8(direction):  # NEW: 'ENE' -> 'E'
  direction = (direction or '').strip().upper()
  if direction not in compass16:
    return None
  return compass8[((compass16.index(direction) + 1) // 2) % 8]

def location_key(loc):          
  text = ' '.join(str(x) for x in loc).lower().replace('ñ', 'n')
  return re.sub(r'[^a-z0-9]', '', text)

def to_bin(field_name, value):  # NEW: numeric value -> label using feature_bins
  if value is None:
    return None
  if field_name == 'wind_dir':
    return value
  for upper, label in feature_bins[field_name]:
    if value <= upper:
      return label
  return feature_bins[field_name][-1][1]


#3. Data loading

def clean_row(row):  # shared clean-up for CSV and JSON rows
  date_value = row.get('date')
  try:
    row['date'] = datetime.datetime.strptime(str(date_value).strip(), '%Y-%m-%d %H:%M:%S')
  except ValueError:
    row['date'] = None
  for field_name in feature:
    row[field_name] = to_number(row.get(field_name))
  row['wind_dir'] = to_compass8(row.get('wind_dir'))
  row['weather'] = (row.get('weather') or '').strip()
  row['weather_type'], row['weather_level'], row['day_night'] = split_weather(row['weather'])
  row['weather_class'] = weather_class(row)
  return row

def tokenize_weather_file(file_path):  # Read one weather data file (CSV-style or JSON) and return a list of dictionaries
  rows = []
  municipality, province = parse_filename(file_path)
  with open(file_path, 'r', encoding='utf-8') as infile:
    content = infile.read().strip()

  if content.startswith('{'):                                          # JSON : {"data": {"field": [values...]}}
    data = json.loads(content).get('data', {})
    n = len(data.get('date', []))
    for i in range(n):
      record = {field_name: (data[field_name][i] if field_name in data and i < len(data[field_name]) else None)
                for field_name in output_fields}
      rows.append(record)
  else:                                                                # CSV : header line + rows
    reader = csv.DictReader(content.splitlines())
    for record in reader:
      row = {}
      for csv_col, value in record.items():
        if csv_col is None:
          continue
        field_name = csv_to_field.get(csv_col.strip())
        if field_name:
          row[field_name] = (value or '').strip()
      rows.append(row)

  cleaned = []
  for row in rows:
    row = clean_row(row)
    if row['date'] is None or row['weather_class'] is None:
      continue
    row['municipality'], row['province'] = municipality, province
    cleaned.append(row)
  return cleaned

def is_weather_file(file):
  return 'MUN_' in file and (file.endswith('.csv') or file.endswith('.json') or '.' not in file)

def in_outlook_folder(root, month_path):
  relative = os.path.relpath(root, month_path)
  return any(part.lower().startswith(outlook_prefix) for part in relative.split(os.sep))

def scan_files(mode):                                     # NEW: list (file_path, (municipality, province)) WITHOUT opening the files
  mode_path = os.path.join(local_drive_path, mode)
  file_list, missing = [], []
  for month in (folder_months.get(mode) or ['']):
    month_path = os.path.join(mode_path, month)
    if not os.path.isdir(month_path):
      missing.append(month_path)
      continue
    for root, dirs, files in os.walk(month_path):
      dirs.sort()                                         # weather-outlook runs in name (time) order
      if use_outlook_folders.get(mode) and not in_outlook_folder(root, month_path):
        continue
      for file in sorted(files):
        if not is_weather_file(file):
          continue
        location = parse_filename(file)
        if location[0]:
          file_list.append((os.path.join(root, file), location))
  return file_list, missing

def load_and_tokenize_data(file_list):  # Load all MUN_ files under directory_path, grouped by (municipality, province)
  tokenized_data = defaultdict(dict)        # {(mun, prov): {date: row}}  (dict by date removes CSV/JSON duplicates)
  file_count = Counter()

  for file_path, location in file_list:
    for row in tokenize_weather_file(file_path):
      tokenized_data[location][row['date']] = row  
    file_count[location] += 1

  grouped = {}
  for location, by_date in tokenized_data.items():
    rows = [by_date[d] for d in sorted(by_date)]
    first_day = rows[0]['date'].date()
    for row in rows:
      row['day_index'] = (row['date'].date() - first_day).days
    grouped[location] = rows
  return grouped, file_count

def WeatherForecast_BOW(tokenized_data):    # Bag of weather codes (count of each weather code across all records)
  weather_dictionary = {}
  for location, rows in tokenized_data.items():
    for row in rows:
      daily_weather = row['weather']
      if daily_weather not in weather_dictionary:
        weather_dictionary[daily_weather] = 1
      else:
        weather_dictionary[daily_weather] += 1
  return weather_dictionary


#4. Learning the Bayesian network (CPTs with Laplace smoothing)

def laplace(counter, values):  # NEW: P(v) = (count(v) + alpha) / (total + alpha * |values|)
  total = sum(counter[v] for v in values)
  return {v: (counter[v] + alpha) / (total + alpha * len(values)) for v in values}

def build_frontier(file_list):  # learn P(DN), P(W | W_prev, DN), P(feature | W) from the observed days
  classes = all_weather_classes
  feature_values = {f: [label for _, label in feature_bins[f]] for f in feature_bins}
  feature_values['wind_dir'] = compass8

  dn_count = Counter()
  trans_count = defaultdict(Counter)
  emission_count = {f: defaultdict(Counter) for f in evidence_features}
  weather_dictionary = Counter()
  record_count = 0

  for i, (file_path, location) in enumerate(file_list, 1):
    rows = tokenize_weather_file(file_path)               # one forecast run for one municipality
    record_count += len(rows)
    for row in rows:                                             # emissions: every observed record
      weather_dictionary[row['weather']] += 1
      dn_count[day_night(row['date'])] += 1
      for f in evidence_features:
        label = to_bin(f, row.get(f))
        if label is not None:
          emission_count[f][row['weather_class']][label] += 1
    slot_rows = [r for r in rows if r['date'].hour % slot_hours == 0]   # transitions: inside the same run only
    for prev, curr in zip(slot_rows, slot_rows[1:]):
      elapsed = (curr['date'] - prev['date']).total_seconds() / 3600
      if elapsed == slot_hours:
        trans_count[f"{prev['weather_class']}|{day_night(curr['date'])}"][curr['weather_class']] += 1
    if i % 1000 == 0:
      print(f"  ... {i} / {len(file_list)} files")

  frontier = {
    'classes': classes,
    'dn_prior': laplace(dn_count, ['D', 'N']),
    'transition': {f"{p}|{dn}": laplace(trans_count[f"{p}|{dn}"], classes) for p in classes for dn in ['D', 'N']},
    'emission': {f: {w: laplace(emission_count[f][w], feature_values[f]) for w in classes} for f in evidence_features},
    'locations': [list(loc) for loc in sorted(tokenized_data)],
    'observed_days': observed_days, 'slot_hours': slot_hours, 'alpha': alpha,
    'transition_counts': {k: dict(v) for k, v in trans_count.items()},
  }
  return frontier, dict(weather_dictionary)

def save_frontier(frontier, frontier_path):
  os.makedirs(os.path.dirname(frontier_path) or '.', exist_ok=True)
  with open(frontier_path, 'w', encoding='utf-8') as outfile:
    json.dump(frontier, outfile, indent=2)

def load_frontier(frontier_path):
  if not os.path.exists(frontier_path):
    return None
  with open(frontier_path, 'r', encoding='utf-8') as infile:
    return json.load(infile)


#5. Evidence for the future slots (no future observations exist -> estimate them from the last 4 days)

def build_evidence(observed_rows, day_index, hour):  # NEW: evidence dict {feature: label} for one future slot
  evidence = {}
  same_hour = [r for r in observed_rows if r['date'].hour == hour]
  if not same_hour:
    return evidence
  for f in evidence_features:
    if f == 'wind_dir':
      values = [r[f] for r in same_hour if r[f]]
      if values:
        evidence[f] = Counter(values).most_common(1)[0][0]
      continue
    points = [(r['day_index'], r[f]) for r in same_hour if r[f] is not None]
    if not points:
      continue
    days = [d for d, _ in points]
    vals = [v for _, v in points]
    mean_day = sum(days) / len(days)
    mean_val = sum(vals) / len(vals)
    estimate = mean_val
    spread = sum((d - mean_day) ** 2 for d in days)
    if use_trend and spread > 0:                                       # least-squares daily trend at this hour
      slope = sum((d - mean_day) * (v - mean_val) for d, v in points) / spread
      estimate = mean_val + slope * (day_index - mean_day)
    if f == 'rain_rate':
      estimate = max(0.0, estimate)
    if f == 'relhum':
      estimate = min(100.0, max(0.0, estimate))
    evidence[f] = to_bin(f, estimate)
  return evidence


#6. Likelihood Weighting

def likelihood_weighting(frontier, belief_prev, dn, evidence, num_samples):
  classes = frontier['classes']
  prev_states, prev_probs = list(belief_prev.keys()), list(belief_prev.values())
  weights = Counter()
  for _ in range(num_samples):
    w = frontier['dn_prior'][dn]                                      # evidence node DN (root): w *= P(dn)
    prev_w = random.choices(prev_states, prev_probs)[0]               # sample W(t-1) from the previous slot's belief
    trans = frontier['transition'][f"{prev_w}|{dn}"]
    curr_w = random.choices(classes, [trans[c] for c in classes])[0]  # sample non-evidence W(t) ~ P(W | W(t-1), DN)
    for f, label in evidence.items():                                 # evidence nodes: w *= P(e_f | W(t))
      w *= frontier['emission'][f][curr_w][label]
    weights[curr_w] += w
  total = sum(weights.values())
  return {c: weights[c] / total for c in classes}                     # normalized P(W(t) | evidence)

def forecast(frontier, rows):  # NEW: run Likelihood Weighting slot by slot for the next forecast_days
  observed_rows = [r for r in rows if r['day_index'] < observed_days]
  last_row = [r for r in observed_rows if r['date'].hour % slot_hours == 0][-1]
  belief = {last_row['weather_class']: 1.0}                           # known weather at the last observed slot
  actual_by_date = {r['date']: r for r in rows if r['day_index'] >= observed_days}
  first_day = rows[0]['date'].replace(hour=0, minute=0, second=0)

  predictions = []
  for day_index in range(observed_days, observed_days + forecast_days):
    for hour in range(0, 24, slot_hours):
      slot_date = first_day + datetime.timedelta(days=day_index, hours=hour)
      dn = day_night(slot_date)
      evidence = build_evidence(observed_rows, day_index, hour)
      belief = likelihood_weighting(frontier, belief, dn, evidence, num_samples)
      actual = actual_by_date.get(slot_date)
      predictions.append({
        'date': slot_date, 'day_index': day_index, 'dn': dn, 'evidence': evidence, 'belief': belief,
        'predicted': max(belief, key=belief.get),
        'p_rain': sum(p for c, p in belief.items() if c.startswith('Rain')),
        'actual': actual['weather_class'] if actual else None,
      })
  return predictions


#7. Reporting

def daily_summary(predictions):  # NEW: Day (06-17) / Night summary per forecast day
  summary = []
  for day_index in sorted({p['day_index'] for p in predictions}):
    for dn in ['D', 'N']:
      part = [p for p in predictions if p['day_index'] == day_index and p['dn'] == dn]
      mean_belief = Counter()
      for p in part:
        for c, prob in p['belief'].items():
          mean_belief[c] += prob / len(part)
      actuals = [p['actual'] for p in part if p['actual']]
      summary.append({
        'date': part[0]['date'].date(), 'dn': 'Day' if dn == 'D' else 'Night',
        'predicted': max(mean_belief, key=mean_belief.get),
        'confidence': max(mean_belief.values()),
        'p_rain': sum(p['p_rain'] for p in part) / len(part),
        'actual': Counter(actuals).most_common(1)[0][0] if actuals else '-',
      })
  return summary

def report(location, predictions, show_actual):
  lines = []
  lines.append(f"\n=== {forecast_days}-day forecast for {location[0]}, {location[1]} "
               f"({num_samples} LW samples per slot, use_trend={use_trend}) ===")
  lines.append(f"{'Slot':<17}{'DN':<4}{'Predicted':<10}{'P(pred)':>8}{'P(rain)':>9}  {'Actual':<8} Evidence")
  for p in predictions:
    ev = ', '.join(f"{k}={v}" for k, v in p['evidence'].items())
    actual = (p['actual'] or '-') if show_actual else ''
    lines.append(f"{p['date'].strftime('%Y-%m-%d %H:%M'):<17}{p['dn']:<4}{p['predicted']:<10}"
                 f"{p['belief'][p['predicted']]:>8.2f}{p['p_rain']:>9.2f}  {actual:<8} {ev}")

  lines.append("\n--- Daily summary ---")
  lines.append(f"{'Date':<12}{'Part':<7}{'Predicted':<10}{'Conf':>6}{'P(rain)':>9}  {'Actual' if show_actual else ''}")
  for s in daily_summary(predictions):
    lines.append(f"{str(s['date']):<12}{s['dn']:<7}{s['predicted']:<10}{s['confidence']:>6.2f}{s['p_rain']:>9.2f}  "
                 f"{s['actual'] if show_actual else ''}")

  if show_actual:
    scored = [p for p in predictions if p['actual']]
    if scored:
      exact = sum(p['predicted'] == p['actual'] for p in scored) / len(scored)
      rain_ok = sum((p['p_rain'] >= 0.5) == p['actual'].startswith('Rain') for p in scored) / len(scored)
      lines.append(f"\nSlots with actual data : {len(scored)}")
      lines.append(f"Exact class accuracy   : {exact:.2%}")
      lines.append(f"Rain / no-rain accuracy: {rain_ok:.2%}")
  return lines

def choose_location(locations):
  keys = [location_key(loc) for loc in locations]
  default_index = keys.index(location_key(default_municipality)) + 1 if location_key(default_municipality) in keys else 1
  candidates = locations
  while len(locations) > 30:  
    text = input(f"\n{len(locations)} municipalities. Type part of a name (blank = {locations[default_index - 1][0]}) : ").strip()
    if not text:
      return locations[default_index - 1]
    candidates = [loc for loc in locations if location_key([text]) in location_key(loc)]
    if candidates:
      break
    print("No match, try again.")
  print("\nMunicipalities found:")
  for i, loc in enumerate(candidates, 1):
    print(f"  [{i}] {loc[0]}, {loc[1]}")
    shown_default = default_index if candidates is locations else 1
    choice = input(f"Choose municipality [default {shown_default}] : ").strip()
    if choice.isdigit() and 1 <= int(choice) <= len(candidates):
      return candidates[int(choice) - 1]
    if candidates is locations:
      return locations[default_index - 1]
    return candidates[0]

def choose_test_location(locations):                     # NEW: 3 pre-programmed test municipalities + option to pick any other
  found = {location_key(loc): loc for loc in locations}
  print("\nTest municipalities:")
  for k, loc in test_municipality.items():
    status = '' if location_key(loc) in found else '   (not in this folder)'
    print(f"  [{k}] {test_municipality_label[k]}{status}")
  other = str(len(test_municipality) + 1)
  print(f"  [{other}] Other municipality from the dataset")
  choice = input("Choose [default 2] : ").strip() or '2'
  if choice in test_municipality:
    loc = found.get(location_key(test_municipality[choice]))
    if loc:
      return loc
    print(f"{test_municipality_label[choice]} is not in this folder. Pick from the list instead.")
  return choose_location(locations)


def choose_start_day(rows, mode):                         
  first_date = rows[0]['date'].date()
  last_index = rows[-1]['day_index']
  latest_start = max(0, last_index - observed_days + 1)  # observation window must fit inside the data
  default_index = 0 if mode == 'testing' else latest_start   # testing: first days (actuals exist after) | implement: newest 4 days
  default_date = first_date + datetime.timedelta(days=default_index)
  print(f"\nData covers {first_date} to {rows[-1]['date'].date()}")
  text = input(f"Start date of the {observed_days}-day observation window (YYYY-MM-DD) [default {default_date}] : ").strip()
  if text:
    try:
      index = (datetime.datetime.strptime(text, '%Y-%m-%d').date() - first_date).days
      if 0 <= index <= latest_start:
        return index
      print(f"Out of range. Latest allowed start is {first_date + datetime.timedelta(days=latest_start)}. Using default.")
    except ValueError:
      print("Invalid date format. Using default.")
  return default_index


#8. Main

def main():
  global local_drive_path

  folder_choice = input("Choose Folder : [1] training | [2] testing | [3] implement : ").strip()
  while folder_choice not in folder_names:
    print("Invalid choice. Please enter '1', '2', or '3'.")
    folder_choice = input("Choose Folder : [1] training | [2] testing | [3] implement : ").strip()
  mode = folder_names[folder_choice]

  dataset_path = os.path.join(local_drive_path, mode)
  if not os.path.isdir(dataset_path):                                  # fall back to the dataset root
    dataset_path = local_drive_path

  tokenized_data, file_count = load_and_tokenize_data(dataset_path)
  if not tokenized_data:
    print(f"\n Not Successful. No data found under {dataset_path}")
    return
  print(f"Loaded {sum(len(r) for r in tokenized_data.values())} weather records "
        f"for {len(tokenized_data)} municipalities from {dataset_path}")

  weather_dictionary = WeatherForecast_BOW(tokenized_data)
  additional_data = []

  if mode == 'training':
    frontier = build_frontier(tokenized_data, file_count)
    save_frontier(frontier, frontier_path)
    additional_data.append(f"Frontier Saved: {frontier_path} ({len(frontier['locations'])} locations)")
    additional_data.append("\n--- P(W(t) | W(t-1), DN) : training transition counts ---")
    for key, counts in sorted(frontier['transition_counts'].items()):
      additional_data.append(f"{key:<10} -> {counts}")
  else:
    frontier = load_frontier(frontier_path)
    if frontier is None:
      print(f"\n Frontier is Empty, Run Training first.")
      return
    print("Frontier loaded.")
    location = choose_location(sorted(tokenized_data))
    predictions = forecast(frontier, tokenized_data[location])
    additional_data.extend(report(location, predictions, show_actual=(mode == 'testing')))

  for line in additional_data:
    print(line)

  try:                                                                 # write the report (BoW + results)
    with open(output_file_path, 'w', encoding='utf-8') as outfile:
      outfile.write(f"Mode : {mode}\n")
      outfile.write(f"Total Provinces:  {len({loc[1] for loc in tokenized_data})}\n")
      outfile.write(f"Total Municipalities:  {len(tokenized_data)}\n")
      outfile.write(f"Total Weather Records:  {sum(len(r) for r in tokenized_data.values())}\n")
      outfile.write(f"Total Weather Codes:  {len(weather_dictionary)}\n\n")
      outfile.write("---WEATHER CODE VOCABULARY---\n")
      for word, count in sorted(weather_dictionary.items()):
        outfile.write(f"{word} : {count}\n")
      outfile.write("\n" + "\n".join(additional_data) + "\n")
    print(f"\nReport written to {output_file_path}")
  except IOError as e:
    print(f"\n NOT SUCCESSFUL. Error writing to file {output_file_path}: {e}")

if __name__ == '__main__':
  main()

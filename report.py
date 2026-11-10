"""Write results as plain comma-separated text files, in the same style as the dataset.

"""
import csv
from datetime import datetime


def write_predictions(src_path, out_path, predictions):
    """Copy the station file row by row and append each method's prediction.

    predictions: {method name: {datetime: predicted label}}. For every method
    two columns are added: the label and whether it matches the Weather column
    ("yes"/"no"). Rows without a prediction get empty cells. The original
    cells (including ranges like 24.9-30.0) are copied unchanged.
    """
    names = list(predictions)
    with open(src_path, newline="") as f:
        rows = [r for r in csv.reader(f) if r]
    header, body = rows[0], rows[1:]
    weather_col = header.index("Weather")
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(header + [c for n in names for c in (n, n + "_Correct")])
        for row in body:
            time = datetime.strptime(row[0].strip(), "%Y-%m-%d %H:%M:%S")
            true = row[weather_col].strip()
            extra = []
            for n in names:
                pred = predictions[n].get(time)
                if pred is None:
                    extra += ["", ""]
                else:
                    extra += [pred, "yes" if pred == true else "no"]
            writer.writerow(row + extra)

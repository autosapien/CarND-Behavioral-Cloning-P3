import csv


def load_data_from_csv(data_file):
    """Load data from a CSV file, also fixes the path issue with the data

    :param data_file: The file to load the data form
    :return:
    """
    # Load data from the file
    rows = []
    prefix_dir = data_file.split('/')[0] + '/IMG/'
    with open(data_file, 'r') as f:
        csv_rows = csv.reader(f)
        next(csv_rows)
        for r in csv_rows:
            rows.append(fix_row_data(r, prefix_dir=prefix_dir))
    return rows


def fix_row_data(row, prefix_dir):
    new_row = [
        prefix_dir + row[0].split('/')[-1],
        prefix_dir + row[1].split('/')[-1],
        prefix_dir + row[2].split('/')[-1],
        float(row[3]),
    ]
    return new_row


def save_data_to_csv(data, data_file):
    """Save data to a CSV file

    :param data: The data to save, a numpy array
    :param data_file: The file to save data to
    :return: None
    """
    with open(data_file, "w") as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        writer.writerow(['c', 'l', 'r', 's'])
        for row in data:
            writer.writerow(row)
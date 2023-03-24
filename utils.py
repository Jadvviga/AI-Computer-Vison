import os
import csv

def choose_file_to_load(folder_path):
    """
    Lists all files in provided folder and asks user to select one of them.
    Returns selected filename.
    """
    print(f"Folder '{folder_path}' contains following files:")
    list_of_files = os.listdir(folder_path)
    for id_, file_name in enumerate(list_of_files):
        print(f"{id_}: '{file_name}'")

    chosen_id = int(input("Choose file id: "))
    return list_of_files[chosen_id]


def model_filename_parse_dimension(file_name):
    """
    Parses a model filename of format ..._dim1_dim2.extension,
    Takes filename as parameter and returns 2 integer tuple of (dim1, dim2)
    """

    extension_removed = file_name.split('.')[-2]
    dimensions = extension_removed.split("_")[-2:]
    dimensions = [int(dim) for dim in dimensions]
    return dimensions


def import_labels(label_file):
    """
    Reads labels file and returns them as dicitonary {"filename": label}
    """

    labels = {}
    with open(label_file) as fd:
        csvreader = csv.DictReader(fd)
        for row in csvreader:
            labels[row['filename']] = int(row['label'])
    return labels

if __name__ == '__main__':
    model_filename_parse_dimension("model_1.sr_model_sth_325_300.hp")

import os
import csv
import matplotlib.pyplot as plt

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
            labels[row['filename']] = int(row['label']) - 1
            # modifies range of classes from 1, 103 to 0,102
    return labels

def make_plots_from_history(history, plots_path, model_filename):
    """
    Plots history of a trained model. Show the plots and saves them to pngs with model_filename
    """
    # Accuracy plot
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.legend(('Training Accuracy', 'Validation accuracy'))
    plt.ylabel('Accuracy')
    plt.xlabel('Number of Epochs')
    
    plot_filename = "plot_accuracy_" + os.path.splitext(model_filename)[0] + ".png"
    plt.savefig(os.path.join(plots_path, plot_filename))
    plt.show()

    # Loss plot
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.legend(('Training Loss', 'Validation Loss'))
    plt.ylabel('Loss')
    plt.xlabel('Number of Epochs')
    
    plot_filename = "plot_loss_" + os.path.splitext(model_filename)[0] + ".png"
    plt.savefig(os.path.join(plots_path, plot_filename))
    plt.show()

if __name__ == '__main__':
    model_filename_parse_dimension("model_1.sr_model_sth_325_300.hp")

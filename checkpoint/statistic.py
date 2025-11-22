import os

def count_files_and_print_parent(directory):
    # Iterate over the items in the directory
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        # Check if the item is a directory
        if os.path.isdir(item_path):
            # Get the list of files in this directory
            files = os.listdir(item_path)
            # Check if there is only one file
            if len(files) == 1:
                # Print the parent directory (the 'None' directory in this case)
                print(f"Parent Directory: {os.path.basename(directory)}")
                print(f"Single file in directory '{item}': {files[0]}")
                print(f"Path: {item_path}")

# Set the directory you want to check
directory_to_check = 'cnn/SEU/FFT'
count_files_and_print_parent(directory_to_check)

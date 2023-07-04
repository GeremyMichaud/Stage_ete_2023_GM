import os

def get_files_in_folder(folder_path):
    try:
        file_names = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
        return file_names
    except OSError as e:
        print(f"Error accessing folder: {e}")
        return []

def remove_extension(file_name, extension):
    if file_name.endswith(extension):
        return file_name[:-len(extension)]
    return file_name

# Replace 'folder_path' with the path to the folder you want to examine
date = "2023-06-27"
path = f"Measurements/{date}/6MV"

files_in_folder = get_files_in_folder(path)
print("Files in the folder:")
for file_name in files_in_folder:
    file_name_without_extension = remove_extension(file_name, '.fit')
    print(file_name_without_extension)

import glob
import os
import pandas as pd
import matplotlib.pyplot as plt

# Specify the folder path where ASC files are located
folder_path = os.path.join("Measurements", "PDD")

# Get a list of ASC files in the specified folder
asc_files = glob.glob(folder_path + '/*.asc')

# Loop through each ASC file
for asc_file in asc_files:
    # Read the content of the ASC file
    with open(asc_file, 'r') as file:
        content = file.readlines()

    # Initialize variables for dataset extraction
    in_dataset = False
    dataset_lines = []
    line_index = 0

    # Loop through each line in the file content using a while loop
    while line_index < len(content):
        line = content[line_index]

        # Check if the line starts with "# Measurement number" indicating the beginning of a dataset
        if line.startswith("# Measurement number"):
            in_dataset = True
            dataset_lines = [line]
        elif in_dataset:
            # Add the line to the dataset_lines list
            dataset_lines.append(line)

            # Check if the line ends with ":EOM  # End of Measurement" indicating the end of a dataset
            if line.endswith(":EOM  # End of Measurement\n"):
                in_dataset = False

                # Extract the name of the original ASC file (without extension)
                file_name = os.path.splitext(os.path.basename(asc_file))[0]

                # Create a new text file for the dataset
                dataset_number = dataset_lines[0].split("\t")[-1].strip()
                output_file_path = os.path.join(folder_path, file_name+"_dataset_"+dataset_number+".txt")

                # Write the dataset content to the new text file
                with open(output_file_path, 'w') as output_file:
                    output_file.writelines(dataset_lines)

        # Increment the line index
        line_index += 1

# Print a message indicating the process is complete
print("Dataset extraction and file creation completed.")

# Read the data from the file
file_path = os.path.join("Measurements", "PDD", "6X_DSP90_5x5_dataset_9.txt")
with open(file_path, 'r') as file:
    lines = file.readlines()

# Initialize lists to store extracted data
data_points = []
dose_data = []

# Open and read the file
with open(file_path, 'r') as file:
    # Iterate through each line
    for line in file:
        # Check if the line starts with "=" indicating a data point
        if line.startswith('='):
            # Split the line into values and convert them to floats
            values = [float(val) for val in line.split()[1:]]  # Skip the first element containing '='
            # Append the values to the data_points list
            data_points.append(values)
            dose_data.append(values[3])


# Initialize lists to store extracted variable and constant axis data
variable_axis_data = []
constant_axis_data = {0: None, 1: None, 2: None}

# Open and read the file
with open(file_path, 'r') as file:
    # Iterate through each line
    for line in file:
        # Check if the line starts with "=" indicating a data point
        if line.startswith('='):
            # Split the line into values and convert them to floats
            values = [float(val) for val in line.split()[1:]]  # Skip the first element containing '='

            # Extract X, Y, Z values
            x_value, y_value, z_value = values[0], values[1], values[2]

            # Check if X values are constant
            if constant_axis_data[0] is None:
                constant_axis_data[0] = x_value
            elif x_value != constant_axis_data[0]:
                constant_axis_data[0] = 'Variable'
            
            # Check if Y values are constant
            if constant_axis_data[1] is None:
                constant_axis_data[1] = y_value
            elif y_value != constant_axis_data[1]:
                constant_axis_data[1] = 'Variable'
            
            # Check if Z values are constant
            if constant_axis_data[2] is None:
                constant_axis_data[2] = z_value
            elif z_value != constant_axis_data[2]:
                constant_axis_data[2] = 'Variable'

# Initialize lists to store extracted variable axis data
variable_data = []

# Open and read the file
with open(file_path, 'r') as file:
    # Iterate through each line
    for line in file:
        # Check if the line starts with "=" indicating a data point
        if line.startswith('='):
            # Split the line into values and convert them to floats
            values = [float(val) for val in line.split()[1:]]  # Skip the first element containing '='

            # Extract X, Y, Z values
            x_value, y_value, z_value = values[0], values[1], values[2]

            # Check if X values are variable
            if constant_axis_data[0] == 'Variable':
                variable_data.append(x_value)
                axis = "X"

            # Check if Y values are variable
            if constant_axis_data[1] == 'Variable':
                variable_data.append(y_value)
                axis = "Y"

            # Check if Z values are variable
            if constant_axis_data[2] == 'Variable':
                variable_data.append(z_value)
                axis = "Z"

plt.plot(variable_data, dose_data)
plt.xlabel(axis)
plt.ylabel("Dose")
plt.show()
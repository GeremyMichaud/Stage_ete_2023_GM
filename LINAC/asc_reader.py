import glob
import os

# Specify the folder path where ASC files are located
folder_path = os.path.join("Measurements", "Ion_chamber")

# Get a list of ASC files in the specified folder
asc_files = glob.glob(os.path.join(folder_path, '*.asc'))

# Initialize variables for dataset extraction
in_dataset = False
dataset_lines = []
dataset_number = None
file_name = None

# Loop through each ASC file
for asc_file in asc_files:
    # Read the content of the ASC file
    with open(asc_file, 'r') as file:
        content = file.readlines()

    # Loop through each line in the file content
    for line in content:
        # Check if the line starts with "# Measurement number" indicating the beginning of a dataset
        if line.startswith("# Measurement number"):
            in_dataset = True
            dataset_lines = [line]
            dataset_number = line.split("\t")[-1].strip()
            file_name = os.path.splitext(os.path.basename(asc_file))[0]
        elif in_dataset:
            # Add the line to the dataset_lines list
            dataset_lines.append(line)

            # Check if the line ends with ":EOM  # End of Measurement" indicating the end of a dataset
            if line.endswith(":EOM  # End of Measurement\n"):
                in_dataset = False

                # Process the dataset content without creating an intermediate file
                data_points = []
                dose_data = []
                constant_axis_data = {0: None, 1: None, 2: None}
                variable_data = []
                first_line_value = None

                for dataset_line in dataset_lines:
                    if dataset_line.startswith('='):
                        values = [float(val) for val in dataset_line.split()[1:]]
                        data_points.append(values)
                        dose_data.append(values[3])
                        x_value, y_value, z_value = values[:3]

                        for i, (axis_value, constant_axis_value) in enumerate(zip((x_value, y_value, z_value), constant_axis_data.values())):
                            if constant_axis_value is None:
                                constant_axis_data[i] = axis_value
                            elif axis_value != constant_axis_value:
                                first_line_value = constant_axis_value
                                variable_data.append(axis_value)
                                axis = "XYZ"[i]

                variable_data.insert(0, first_line_value)

                # Determine the directory based on the axis
                directory = os.path.join(folder_path, "Profile" if axis in ("X", "Y") else "PDD")
                os.makedirs(directory, exist_ok=True)

                # Write the processed data directly to the output file
                output_file_path = os.path.join(directory, f"{file_name}_dataset_{dataset_number}_{axis}.txt")
                with open(output_file_path, 'w') as output_file:
                    output_file.write("Axis\tDose\n")
                    for axis, dose in zip(variable_data, dose_data):
                        output_file.write(f"{axis}\t{dose}\n")

# Print a message indicating the process is complete
print("Dataset extraction and file creation completed.")

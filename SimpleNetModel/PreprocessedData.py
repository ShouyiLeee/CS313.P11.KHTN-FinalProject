import os
import shutil

def merge_and_rename_images(source_dir, target_dir):
    # Create the target directory if it doesn't exist
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # Iterate through the subfolders in the source directory
    for subfolder in os.listdir(source_dir):
        subfolder_path = os.path.join(source_dir, subfolder)
        if os.path.isdir(subfolder_path) and subfolder != 'good':
            # Iterate through the files in the subfolder
            for filename in os.listdir(subfolder_path):
                file_path = os.path.join(subfolder_path, filename)
                if os.path.isfile(file_path):
                    base, ext = os.path.splitext(filename)
                    new_filename = f"{subfolder}_{base}{ext}"
                    new_file_path = os.path.join(target_dir, new_filename)
                    shutil.move(file_path, new_file_path)
                    print(f"Moved {file_path} to {new_file_path}")

# Example usage
source_dir = 'D:/IT/GITHUB/CS313.P11.KHTN-FinalProject/SimpleNet/metal_nut/test'
target_dir = 'D:/IT/GITHUB/CS313.P11.KHTN-FinalProject/SimpleNet/metal_nut/test/Anomaly'
merge_and_rename_images(source_dir, target_dir)
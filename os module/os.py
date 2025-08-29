# ---------------------------------
# Python OS Module Functions
# ---------------------------------
import os

# Important functions of the Python os module:
# 1. Handling the Current Working Directory
# 2. Creating a Directory
# 3. Listing Files and Directories
# 4. Deleting Directories or Files
# 5. File Permissions and Metadata (check existence, rename, etc.)

# -------------------------------
# 1. Handling the Current Working Directory
# -------------------------------
path = os.getcwd()
print("Current Directory:", path)
# Example: C:\Users\Tareq\Desktop

# Change to parent directory
os.chdir('../')
new_path = os.getcwd()
print("After changing:", new_path)
# Example: C:\Users\Tareq

# -------------------------------
# 2. Creating a Directory
# -------------------------------

# mkdir() → create single directory
current_path = os.getcwd()
directory = "E"
path = os.path.join(current_path, directory)
os.mkdir(path)   # Creates folder "E" inside current directory

# makedirs() → create nested directories
current_path = os.getcwd()
directory = "Esc/SubFolder"
path = os.path.join(current_path, directory)
os.makedirs(path)   # Creates Esc folder with SubFolder inside it

# -------------------------------
# 3. Listing Files & Directories
# -------------------------------
items = os.listdir()
print("Items in directory:", items)
# Example: ['E', 'Esc', 'file1.txt', 'script.py']

# -------------------------------
# 4. Deleting Directory or Files
# -------------------------------

# Remove a file
current_path = os.getcwd()
file_to_delete = os.path.join(current_path, "file1.txt")
# os.remove(file_to_delete)   # ⚠️ Deletes file1.txt permanently

# Remove a folder (only if empty)
folder_to_delete = os.path.join(current_path, "E")
# os.rmdir(folder_to_delete)  # ⚠️ Deletes folder E if empty

# -------------------------------
# 5. Rename Folder or File
# -------------------------------
current_path = os.getcwd()
path_old = os.path.join(current_path, "E")
path_new = os.path.join(current_path, "G")
# os.rename(path_old, path_new)   # Renames folder E → G

# -------------------------------
# 6. Check if File or Folder Exists
# -------------------------------
path = os.getcwd()
if os.path.exists(path):
    print("Path exists ✅")
else:
    print("Path does not exist ❌")
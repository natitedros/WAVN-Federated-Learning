import os

folder_path = "./edge_detection_results_1000_01262026/"  # change this

for filename in os.listdir(folder_path):
    if filename.endswith(".png_hed.png"):
        old_path = os.path.join(folder_path, filename)
        new_filename = filename.replace(".png_hed.png", "_hed.png", 1)
        new_path = os.path.join(folder_path, new_filename)
        os.rename(old_path, new_path)

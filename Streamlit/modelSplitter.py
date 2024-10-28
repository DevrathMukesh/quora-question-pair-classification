import os

# Define model path and part size (e.g., 10 MB per part)
model_path = 'models/RandomForest_Bow.pkl'
part_size = 20 * 1024 * 1024  # 10 MB in bytes

# Read and split the file
with open(model_path, 'rb') as file:
    part_num = 0
    chunk = file.read(part_size)
    while chunk:
        with open(f'models/RandomForest_Bow_part{part_num}.pkl', 'wb') as part_file:
            part_file.write(chunk)
        part_num += 1
        chunk = file.read(part_size)

print(f"Model split into {part_num} parts.")


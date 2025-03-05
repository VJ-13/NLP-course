import os
import re

def split_by_chapters(input_file, output_folder):
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Read the entire file
    with open(input_file, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # Use regex to detect chapter headings
    chapters = re.split(r'(?=\bChapter\s+\d+\b)', content, flags=re.IGNORECASE)
    
    # Save each chapter as a separate file
    for i, chapter in enumerate(chapters):
        if chapter.strip():  # Ignore empty sections
            chapter_file = os.path.join(output_folder, f'Book_5-Chapter_{i}.txt')
            with open(chapter_file, 'w', encoding='utf-8') as out_file:
                out_file.write(chapter.strip())
    
    print(f"Successfully split into {len(chapters)} files in '{output_folder}'.")

# Example usage
input_filename = "Percy_Jackson-5.txt"  # Change this to your input file
output_directory = "Percy-Jackson"  # Folder where split files will be stored
split_by_chapters(input_filename, output_directory)

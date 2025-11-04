import os
import pandas as pd

# --- Configuration ---
# Define the directories where your email files are stored
HAM_DIR = 'C:\\Users\\jappa\\Repos\\cs324\\hard_ham'
SPAM_DIR = 'C:\\Users\\jappa\\Repos\\cs324\\spam'
OUTPUT_CSV = 'emails_dataset.csv'
# ---------------------

def load_emails_from_folder(folder_path, label):
    """
    Reads all text files from a specified folder and creates a list
    of dictionaries, each containing the email text and its label.
    """
    emails = []
    
    # Check if the folder exists
    if not os.path.isdir(folder_path):
        print(f"Warning: Directory '{folder_path}' not found. Skipping.")
        return emails

    # Iterate through all files in the directory
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        
        # Ensure it's a file and not a directory
        if os.path.isfile(file_path):
            try:
                # Open the file and read its entire content
                # 'latin-1' or 'ignore' is often necessary for older email datasets
                # which can contain various encoding issues.
                with open(file_path, 'r', encoding='latin-1') as f:
                    content = f.read()
                
                # Append the label and the full text content
                emails.append({'label': label, 'text': content})
                
            except Exception as e:
                # Print an error but continue processing other files
                print(f"Error reading file {filename} in {folder_path}: {e}")
                
    print(f"Successfully loaded {len(emails)} emails from the '{folder_path}' directory.")
    return emails

# --- Main Execution ---

if __name__ == "__main__":
    
    # 1. Load Ham Emails
    ham_data = load_emails_from_folder(HAM_DIR, 'ham')
    
    # 2. Load Spam Emails
    spam_data = load_emails_from_folder(SPAM_DIR, 'spam')
    
    # 3. Combine the data
    all_data = ham_data + spam_data
    
    if not all_data:
        print("\nNo emails were loaded. Please check your folder names and paths.")
    else:
        # 4. Create a Pandas DataFrame
        df = pd.DataFrame(all_data)
        
        # Optional: Clean the 'text' column by replacing newlines with spaces
        # This makes the CSV easier to handle, but may or may not be desired
        # depending on your classification method. I'll leave it as is for raw text.
        
        # 5. Save the DataFrame to a CSV file
        df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8')
        
        print(f"\n✅ Data collection complete!")
        print(f"Total emails processed: {len(df)}")
        print(f"Output saved to: {OUTPUT_CSV}")
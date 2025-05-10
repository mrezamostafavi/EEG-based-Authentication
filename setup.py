from google.colab import drive

def initialize_environment():
    print("Mounting Google Drive...")
    drive.mount('/gdrive')
    print("Navigating to working directory...")
    %cd /gdrive
    print("Setup complete.")

if __name__ == "__main__":
    initialize_environment()

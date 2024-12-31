import cv2
import os
import json

# JSON file to store classified filenames
json_file = "/media/admin-anedunga/ASHWIN-128/classified_depths.json"

def load_classified_data(json_file):
    """Load existing classifications from the JSON file or create a new structure."""
    if os.path.exists(json_file):
        with open(json_file, 'r') as f:
            return json.load(f)
    else:
        return {"good": [], "bad": []}

def save_classified_data(json_file, data):
    """Save the classified data to the JSON file."""
    with open(json_file, 'w') as f:
        json.dump(data, f, indent=4)

def main():
    # Directory containing depth map images
    dir = "/media/admin-anedunga/ASHWIN-128/Navvis_Labeling/"

    # Load existing classified data
    classified_data = load_classified_data(json_file)

    # Get the list of already classified files
    classified_files = set(classified_data["good"] + classified_data["bad"])

    # Iterate through the images in the directory
    for file in os.listdir(dir):
        if file.endswith(('.png', '.jpg', '.jpeg')) and file not in classified_files:
            file_path = os.path.join(dir, file)
            
            # Load and process the image
            image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                print(f"Error loading image: {file}")
                continue

            # Downsize the image to 1/3 of its original size
            resized_image = cv2.resize(image, (768, 968), interpolation=cv2.INTER_NEAREST)

            # Apply the inferno colormap
            colored_image = cv2.applyColorMap(resized_image, cv2.COLORMAP_INFERNO)

            # Display the image
            cv2.imshow("Depth Map", colored_image)
            print(f"Classify image: {file} (Press 'G' for Good, 'B' for Bad, 'Q' to Quit)")

            while True:
                key = cv2.waitKey(0) & 0xFF
                if key == ord('g'):
                    classified_data["good"].append(file)
                    print(f"{file} classified as Good.")
                    break
                elif key == ord('b'):
                    classified_data["bad"].append(file)
                    print(f"{file} classified as Bad.")
                    break
                elif key == ord('q'):
                    print("Exiting classification.")
                    save_classified_data(json_file, classified_data)
                    cv2.destroyAllWindows()
                    return

            # Save the data to the JSON file after each classification
            save_classified_data(json_file, classified_data)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

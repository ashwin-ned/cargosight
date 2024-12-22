import csv
class BenchmarkLogger:
    """ This class receives the metric scores and logs them with ; delimitation to a .csv file."""
    def __init__(self, image_name, folder_name, model_name, metadata, inference_time, mean_absolute_error, root_mean_squared_error, absolute_relative_error, threshold_accuracy):
        self.image_name = image_name
        self.folder_name = folder_name
        self.model_name = model_name
        self.metadata = metadata
        self.inference_time = inference_time
        self.mean_absolute_error = mean_absolute_error
        self.root_mean_squared_error = root_mean_squared_error
        
        self.absolute_relative_error = absolute_relative_error
        self.threshold_accuracy = threshold_accuracy
        self.entries = []

    def add_entry(self):
        """ Adds an entry to the log with the provided metrics and metadata."""
    
        entry = {
            'image_name': self.image_name,
            'folder_name': self.folder_name,
            'model_name': self.model_name,
            'metadata': self.metadata,
            'inference_time': self.inference_time, 
            'mean_absolute_error': self.mean_absolute_error,
            'root_mean_squared_error': self.root_mean_squared_error,
            'absolute_relative_error': self.absolute_relative_error,
            'threshold_accuracy': self.threshold_accuracy
        }
        self.entries.append(entry)

    def save_to_csv(self):
        """ Saves the logged entries to a CSV file with ';' as the delimiter. """
        with open(f"{self.model_name}_benchmark_log.csv", mode='a', newline='') as file:
            writer = csv.DictWriter(file, delimiter=';', fieldnames=[
                'image_name', 'folder_name', 'model_name', 'metadata', 'inference_time',
                'mean_absolute_error', 'root_mean_squared_error', 'absolute_relative_error', 'threshold_accuracy'
            ])
            #writer.writeheader()
            for entry in self.entries:
                writer.writerow(entry)
        file.close()

# Example usage:
# logger = BenchmarkLogger('benchmark_log.csv', mean_absolute_error, root_mean_squared_error, absolute_relative_error)
# logger.add_entry('image1.png', 'depth1.png', {'size': '640x480'}, 0.1, 2.5, 0.05)
# logger.add_entry('image2.png', 'depth2.png', {'size': '640x480'}, 0.08, 2.3, 0.04)
# logger.save_to_csv()
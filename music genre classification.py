import os
import librosa
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


Genre = ['blues', 'classical', 'country', 'disco', 'hiphop', 'metal', 'pop', 'rock']

# Load and extract MFCC features
def load_data(dataset_path):
    features, labels = [], []
    for genre in Genre:
        genre_folder = os.path.join(dataset_path, genre)
        for file in os.listdir(genre_folder):
            if file.endswith(".wav"):
                audio_path = os.path.join(genre_folder, file)
                try:
                    signal, sr = librosa.load(audio_path, sr=None)
                    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=10)
                    mfcc_features = np.mean(mfcc, axis=1)
                    features.append(mfcc_features)
                    labels.append(genre)
                except Exception as e:
                    print(f"Error processing file {audio_path}: {e}")
    return np.array(features), np.array(labels)

# Split data into training and testing sets
def split_data(features, labels, test_size=0.2):
    indices = np.arange(len(features))
    np.random.shuffle(indices)
    split_point = int(len(features) * (1 - test_size))
    train_indices, test_indices = indices[:split_point], indices[split_point:]
    return features[train_indices], labels[train_indices], features[test_indices], labels[test_indices]

# Normalize features
def normalize_features(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    return (X - mean) / std, mean, std

# KNN classifier
def knn_predict_probabilities(X_train, y_train, X_test, k=3):
    probabilities = []
    for test_sample in X_test:
        distances = np.linalg.norm(X_train - test_sample, axis=1)
        k_indices = np.argsort(distances)[:k]
        k_neighbors = y_train[k_indices]
        genre_count = {genre: 0 for genre in Genre}
        for neighbor in k_neighbors:
            genre_count[neighbor] += 1
        probabilities.append({genre: count / k for genre, count in genre_count.items()})
    return probabilities

# Predict genre and display probabilities
def predict_genre():
    try:
        status_label.config(text="Status: Prediction in progress...", fg="blue")
        file_path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav")])
        if not file_path:
            status_label.config(text="Status: No file selected", fg="red")
            return

        # Load and preprocess the audio file
        signal, sr = librosa.load(file_path, sr=None)
        mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=10)
        new_song_features = np.mean(mfcc, axis=1)
        new_song_features = (new_song_features - train_mean) / train_std

        # Predict genre probabilities
        new_song_probabilities = knn_predict_probabilities(X_train, y_train, [new_song_features], k=3)
        probabilities = new_song_probabilities[0]

        # Display probabilities in a bar chart
        Genre = list(probabilities.keys())
        probabilities_values = list(probabilities.values())
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(Genre, probabilities_values, color='skyblue')
        ax.set_title('Genre Probabilities')
        ax.set_ylabel('Probability')
        ax.set_xlabel('Genres')
        plt.tight_layout()

        # Show plot in Tkinter window
        canvas = FigureCanvasTkAgg(fig, master=plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack()

        status_label.config(text="Status: Prediction complete", fg="green")

    except Exception as e:
        messagebox.showerror("Error", str(e))
        status_label.config(text="Status: Error occurred", fg="red")

# clear the bar chart
def clear_plot():
    for widget in plot_frame.winfo_children():
        widget.destroy()
    status_label.config(text="Status: Ready", fg="black")

print("Loading GTZAN dataset...")
dataset_path = r"C:\Users\Somu C\Desktop\genres_original" 
X, y = load_data(dataset_path)
X_train, y_train, X_test, y_test = split_data(X, y, test_size=0.3)
X_train, train_mean, train_std = normalize_features(X_train)
X_test = (X_test - train_mean) / train_std

root = tk.Tk()
root.title("Music Genre Predictor")
root.geometry("800x600")

input_frame = tk.Frame(root)
input_frame.pack(pady=10)

plot_frame = tk.Frame(root)
plot_frame.pack(pady=10)

control_frame = tk.Frame(root)
control_frame.pack(pady=10)

predict_button = tk.Button(input_frame, text="Select Audio File", command=predict_genre)
predict_button.grid(row=0, column=0, padx=5)

clear_button = tk.Button(control_frame, text="Clear Plot", command=clear_plot)
clear_button.pack()
status_label = tk.Label(root, text="Status: Ready", fg="black")
status_label.pack(pady=10)

root.mainloop()

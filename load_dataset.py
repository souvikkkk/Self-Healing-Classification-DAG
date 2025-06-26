from datasets import load_dataset

# Load IMDb sentiment dataset
dataset = load_dataset("imdb")

# Show sample
print("Sample training example:")
print(dataset["train"][0])

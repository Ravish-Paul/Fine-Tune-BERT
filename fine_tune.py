import os
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    pipeline
)

def generate_dummy_data():
    """Generates a dummy sentiment dataset and saves it as sample_data.csv."""
    print("Generating dummy sentiment dataset...")
    data = {
        "text": [
            "I love this movie",
            "This product is amazing",
            "Fantastic customer service",
            "The food was delicious",
            "I am very happy with the purchase",
            "Great quality and fast delivery",
            "Excellent experience overall",
            "This app is very useful",
            "Highly recommended",
            "The book was interesting",
            "I enjoyed every moment",
            "Amazing performance",
            "The staff was friendly",
            "Best purchase ever",
            "Wonderful experience",
            "I am satisfied",
            "Very good product",
            "Works perfectly",
            "Outstanding quality",
            "Loved it",
            "I hate this service",
            "Very bad experience",
            "The product is terrible",
            "Waste of money",
            "Not worth buying",
            "Poor customer support",
            "I am disappointed",
            "Worst experience ever",
            "The app keeps crashing",
            "The food was awful",
            "Bad quality",
            "Completely useless",
            "I regret buying this",
            "Very frustrating",
            "The book was boring",
            "The staff was rude",
            "Delivery was late",
            "Not satisfied",
            "Terrible performance",
            "It stopped working",
            "Good value for money",
            "Happy with the service",
            "The design is beautiful",
            "Very reliable product",
            "The camera quality is excellent",
            "Poor battery life",
            "The screen is broken",
            "Customer service was unhelpful",
            "The software is buggy",
            "Excellent battery backup"
        ],
        "label": [
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            1, 1, 1, 1, 1,
            0, 0, 0, 0,
            1
        ]
    }
    df = pd.DataFrame(data)
    df.to_csv("sample_data.csv", index=False)
    print("Dataset saved to sample_data.csv")

def main():
    # 1. Generate or check dataset
    csv_file = "sample_data.csv"
    if not os.path.exists(csv_file):
        generate_dummy_data()
        
    # 2. Load dataset
    print("Loading dataset...")
    dataset = load_dataset("csv", data_files=csv_file)
    print("Dataset structure:")
    print(dataset)

    # 3. Train-test split
    print("Splitting dataset...")
    dataset = dataset["train"].train_test_split(test_size=0.2, seed=42)
    print(dataset)
    print("Final training dataset structure:")
    print(dataset["train"])

    # 4. Load model and tokenizer
    model_name = "bert-base-uncased"
    print(f"Loading pretrained model and tokenizer: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2
    )

    # 5. Tokenization
    print("Tokenizing dataset...")
    def tokenise(example):
        return tokenizer(
            example["text"],
            padding="max_length",
            truncation=True,
        )
    
    tokenized_data = dataset.map(tokenise)

    # 6. Training Configuration
    print("Setting up Trainer...")
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=10,
        per_device_train_batch_size=8
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_data["train"]
    )

    # 7. Fine-Tuning
    print("Starting training...")
    trainer.train()
    print("Training completed.")

    # 8. Save model and tokenizer
    model_dir = "sentiment_model"
    print(f"Saving fine-tuned model to {model_dir}...")
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    print("Model saved successfully.")

    # 9. Testing Inference
    print("\nRunning inference tests on the saved model...")
    classifier = pipeline(
        "text-classification",
        model=model_dir,
        tokenizer=model_dir
    )

    def predict(text):
        result = classifier(text)[0]
        sentiment = "Positive 😊" if result["label"] == "LABEL_1" else "Negative 😞"
        print(f"Text: {text}")
        print(f"Sentiment: {sentiment}")
        print(f"Confidence: {round(result['score'], 4)}")
        print("-" * 40)

    # Run predictions from Colab notebook
    predict("This product is fantastic")
    predict("I hate this product")
    predict("Amazing quality")
    predict("Worst purchase ever")

if __name__ == "__main__":
    main()

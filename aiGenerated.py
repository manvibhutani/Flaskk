from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import os

# Set up paths and load the model
def load_model():
    absolute_path = os.path.dirname(__file__)
    relative_path = "./roberta/"
    full_path = os.path.join(absolute_path, relative_path)

    tokenizer = AutoTokenizer.from_pretrained("FacebookAI/roberta-base")
    model = AutoModelForSequenceClassification.from_pretrained("FacebookAI/roberta-base")
    classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
    return classifier

# Define the function to get the percentage of AI-generated content
def classify_text(text):
    classifier = load_model()
    if text:
        res = classifier(text, truncation=True, max_length=510)
        label = res[0]['label']
        score = res[0]['score']

        if label == 'Real':
            real_score = score * 100
            fake_score = 100 - real_score
        else:
            fake_score = score * 100
            real_score = 100 - fake_score

        return {
            "AI-generated_percentage": fake_score,
            "Human-generated_percentage": real_score
        }
    else:
        return {
            "AI-generated_percentage": 50,
            "Human-generated_percentage": 50
        }

# Example usage
text = """In a distant future, where cities floated above the clouds, a young engineer named Lyra discovered an ancient map hidden in the depths of an abandoned library. The map depicted a forgotten world, far below the towering cloud cities, where lush forests and sparkling rivers once thrived.

Driven by curiosity and a longing for adventure, Lyra set out on a quest to reach this mysterious land beneath the clouds. She built a small airship, crafted from forgotten technologies and powered by an energy source no one in her generation understood. As she descended through the misty layers of clouds, the light above faded, and a new world began to emerge beneath her.

It was a land of untamed beauty, where nature had reclaimed the earth. Towering trees with glowing leaves stretched toward the sky, and creatures of legend roamed freely. Lyra marveled at the vibrant life that had been hidden for centuries.

But something stirred in the shadows of the forests—an ancient force that had slumbered for eons. It watched Lyra’s every move, curious yet wary of the intruder. As she ventured deeper into this forgotten world, Lyra would soon uncover the truth of the past, and her place in the future of both worlds above and below the clouds.

"""
result = classify_text(text)
print(f"AI-generated: {result['AI-generated_percentage']:.2f}%")
print(f"Human-generated: {result['Human-generated_percentage']:.2f}%")

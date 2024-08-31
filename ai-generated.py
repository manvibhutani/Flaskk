from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification, logging
import base64
import io
import docx2txt
import PyPDF2
import os

# -----------------------------------

def parse_contents(content, filename):
    content_type, content_string = content.split(',')
    decoded = base64.b64decode(content_string)
    
    try:
        if filename.endswith('.docx'):
            text = docx2txt.process(io.BytesIO(decoded))
        elif filename.endswith('.txt'):
            text = decoded.decode()
        elif filename.endswith('.pdf'):
            reader = PyPDF2.PdfReader(io.BytesIO(decoded))
            text = ''
            for page in reader.pages:
                text += page.extract_text()
        else:
            raise ValueError("Unsupported file format")
    except Exception as e:
        print(e)
        raise Exception("Error processing file")

    return text

class AIDetector:
    def __init__(self):
        logging.set_verbosity_error()
        absolute_path = os.path.dirname(__file__)
        relative_path = "./roberta-base-openai-detector/"
        full_path = os.path.join(absolute_path, relative_path)
        tokenizer = AutoTokenizer.from_pretrained(full_path, local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(full_path, local_files_only=True)
        self.classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)

    def classify_text(self, text):
        if text and text.strip():
            res = self.classifier(text, truncation=True, max_length=510)
            label = res[0]['label']
            score = res[0]['score']

            if label == 'AI-Generated':
                ai_score = score * 100
                human_score = 100 - ai_score
                return {
                    "human_score": human_score,
                    "ai_score": ai_score,
                    "human_score_label": f"{human_score:.0f}%",
                    "ai_score_label": f"{ai_score:.0f}%"
                }
            else:
                human_score = score * 100
                ai_score = 100 - human_score
                return {
                    "human_score": human_score,
                    "ai_score": ai_score,
                    "human_score_label": f"{human_score:.0f}%",
                    "ai_score_label": f"{ai_score:.0f}%"
                }
        else:
            return {
                "human_score": 50,
                "ai_score": 50,
                "human_score_label": "",
                "ai_score_label": ""
            }

# Example usage
if __name__ == '__main__':
    # Example base64 encoded content and filename
    example_content = "data:application/pdf;base64,..."
    example_filename = "fake1.pdf"
    
    try:
        text = parse_contents(example_content, example_filename)
        detector = AIDetector()
        results = detector.classify_text(text)
        print("Results:", results)
    except Exception as e:
        print("Error:", str(e))

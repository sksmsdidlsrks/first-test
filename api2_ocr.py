from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from transformers import BertTokenizer, BertForSequenceClassification
from PIL import Image
import requests
import torch

# TrOCR 모델과 프로세서 로드
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed")

# BERT 모델과 토크나이저 로드 (텍스트 분류용)
classifier_model = BertForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
tokenizer = BertTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")

# 이미지 URL (올바른 URL 사용)
image_path = "기차표.jpg"  # 여기에 실제 이미지 URL로 변경
image = Image.open(image_path)

# 이미지 전처리
pixel_values = processor(image, return_tensors="pt").pixel_values

# 텍스트 생성
generated_ids = model.generate(pixel_values, max_new_tokens=50)
generated_text = processor.decode(generated_ids[0], skip_special_tokens=True)

# 텍스트 분류 함수
def classify_text(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        logits = classifier_model(**inputs).logits
    predicted_class = logits.argmax().item()
    return predicted_class

# 추출된 텍스트 분류
category = classify_text(generated_text)

# 결과 출력
print("추출된 텍스트:", generated_text)
print("분류된 카테고리:", category)

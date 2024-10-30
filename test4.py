import requests
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM, BartForConditionalGeneration, BartTokenizer
import os

# 장치 설정
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# 모델과 프로세서 초기화
model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-large", torch_dtype=torch_dtype, trust_remote_code=True).to(device)
processor = AutoProcessor.from_pretrained("microsoft/Florence-2-large", trust_remote_code=True)

# 요약 모델과 토크나이저 초기화
summarizer_model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn").to(device)
summarizer_tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")

# 이미지 파일 경로 설정
image_folder = r"img"  # raw string으로 경로 설정
image_paths = [os.path.join(image_folder, img) for img in os.listdir(image_folder) if img.endswith(('.jpg', '.png'))]

# 선택한 이미지 경로 출력 및 처리
for img_path in image_paths:
    print(f"Processing image: {img_path}")
    
    # 이미지 열기
    image = Image.open(img_path).convert("RGB")  # RGB 모드로 변환
    image = image.resize((224, 224))  # 필요시 이미지 크기 조정

    # 프롬프트 설정
    prompt = "<OD>"

    # 입력 처리
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device, torch_dtype)

    # 텍스트 생성
    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        num_beams=3,
        do_sample=False
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    # 텍스트 요약
    summary_inputs = summarizer_tokenizer(generated_text, return_tensors="pt", max_length=1024, truncation=True).to(device)
    summary_ids = summarizer_model.generate(summary_inputs["input_ids"], max_length=150, min_length=30, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary_text = summarizer_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    # 결과를 한글로 출력하기 위해 번역 모델 사용 (예: MarianMT)
    from transformers import MarianMTModel, MarianTokenizer
    
    # 한글로 번역하기 위한 모델과 토크나이저 초기화
    translation_model_name = "Helsinki-NLP/opus-mt-en-ko"
    translation_model = MarianMTModel.from_pretrained(translation_model_name).to(device)
    translation_tokenizer = MarianTokenizer.from_pretrained(translation_model_name)

    # 생성된 텍스트 번역
    translated_generated_inputs = translation_tokenizer(generated_text, return_tensors="pt", padding=True, truncation=True).to(device)
    translated_generated_ids = translation_model.generate(translated_generated_inputs["input_ids"], max_length=150)
    translated_generated_text = translation_tokenizer.decode(translated_generated_ids[0], skip_special_tokens=True)

    # 요약 텍스트 번역
    translated_summary_inputs = translation_tokenizer(summary_text, return_tensors="pt", padding=True, truncation=True).to(device)
    translated_summary_ids = translation_model.generate(translated_summary_inputs["input_ids"], max_length=150)
    translated_summary_text = translation_tokenizer.decode(translated_summary_ids[0], skip_special_tokens=True)

    # 결과 출력
    print("Generated Text (Korean):")
    print(translated_generated_text)
    print("\nSummary (Korean):")
    print(translated_summary_text)
    print("\n" + "="*40 + "\n")  # 구분선

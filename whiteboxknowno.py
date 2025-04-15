!git clone https://github.com/cog-model/AmbiK-dataset
!pip install -r /content/AmbiK-dataset/requirements.txt

from huggingface_hub import notebook_login
notebook_login()

!cp "/content/AmbiK-dataset/utils/metrics.py" "/content/"

import sys
sys.path.append('/content/AmbiK-dataset/utils/metrics.py')

from metrics import batch_metric_calculation
import pandas as pd
import wandb
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from scipy.stats import entropy
from tqdm import tqdm
import wandb
import matplotlib.pyplot as plt
import os
import json
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
import torch.nn.functional as F
import re
from sklearn.metrics import f1_score
from sklearn.linear_model import LinearRegression

wandb.login(key='33780493ec2591d3936708b76fe1e229e35139b8')

calibration_data = pd.read_csv("/content/AmbiK-dataset/ambik_dataset/ambik_calib_100.csv")
test_data = pd.read_csv("/content/AmbiK-dataset/ambik_dataset/ambik_test_400.csv")

calibration_data = calibration_data[['environment_short','amb_shortlist','question', 'ambiguity_type','user_intent']]
test_data = test_data[['environment_short','amb_shortlist','question', 'ambiguity_type','user_intent']]

wandb.init(project="llm-WhiteBox-KnowNo",
          config={
              "model": "Llama-3.1-8B-Instruct",
              "detection_method": "White-box + KnowNo",
              "n_samples": 4,
              "temperature": 0.7,
              "max_new_tokens": 30
    }
)

wandb.config.update({
    "test_samples": len(test_data),
    "ambiguity_types": list(test_data['ambiguity_type'].unique())
})

model_id = "meta-llama/Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto",
    output_attentions=True,
    output_hidden_states=True,
    attn_implementation="eager",
)

test_data['amb_shortlist'] = test_data['amb_shortlist'].fillna(' ')

import re

def clean_llm_answers(answer: str) -> list[str]:
    """Извлекает ответы модели из текста, удаляя вопросы и окружение."""
    if not isinstance(answer, str):
        return [""]  # Возвращаем список с пустой строкой вместо пустого списка

    # Удаляем блоки с Environment и вопросами
    answer = re.sub(
        r'(Question:|Evironment:|Environment:|System:|User:|Пользователь:|Context:|Контекст:|Prompt:|Инструкция:).*?(\n|$)',
        '',
        answer,
        flags=re.IGNORECASE | re.DOTALL
    ).strip()

    # Если после очистки остался только ответ без префикса "Answer:"
    if answer and not re.search(r'^(Answer:|Ответ:)', answer, re.IGNORECASE):
        return [answer]

    # Ищем основной ответ (после "Answer:")
    answer_match = re.search(
        r'(?:Answer:|Ответ:)\s*(.*)',
        answer,
        flags=re.IGNORECASE | re.DOTALL
    )

    if answer_match:
        answer_text = answer_match.group(1).strip()
        # Удаляем нумерацию и лишние символы
        answer_text = re.sub(r'^[\d\s]*[\.\-\)\*\+]\s*', '', answer_text, flags=re.MULTILINE)
        # Удаляем кавычки если они окружают весь текст
        if (answer_text.startswith(('"', "'", "«")) and answer_text.endswith(('"', "'", "»"))):
            answer_text = answer_text[1:-1].strip()
        # Нормализуем пробелы
        answer_text = re.sub(r'\s+', ' ', answer_text).strip()

        return [answer_text] if answer_text else [""]

    return [""]  # Всегда возвращаем список с хотя бы пустой строкой

class AmbiguityDetector:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.sbert = SentenceTransformer('all-MiniLM-L6-v2')
        self.model.config.pad_token_id = self.tokenizer.eos_token_id

    def prompt_analyze(self, prompt: str, n_samples=4)->dict:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        generated_texts = []
        attention_dispersions = []
        hidden_dispersions = []

        max_len = 0

        for _ in range(n_samples):
          with torch.no_grad():
              output = self.model.generate(
                    **inputs,
                    max_new_tokens=30,
                    do_sample=True,
                    temperature=0.7,
                    output_attentions=True,
                    return_dict_in_generate=True,
                    output_hidden_states=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )

          generated_texts.append(self.tokenizer.decode(output.sequences[0], skip_special_tokens=True))

          if output.attentions is not None and len(output.attentions) > 0:
                last_layer_attentions = []
                for layer_attn in output.attentions:
                    if isinstance(layer_attn, tuple):
                        layer_attn = layer_attn[0]

                    max_len = max(max_len, layer_attn.shape[-1])
                    last_layer_attentions.append(layer_attn[0, :, -1, :])

                padded_attention_tensors = []
                for attn in last_layer_attentions:
                  if attn.shape[-1] < max_len:
                    padding_size = max_len - attn.shape[-1]
                    padded_attn = F.pad(attn, (0, padding_size))
                    padded_attention_tensors.append(padded_attn)
                  else:
                    padded_attention_tensors.append(attn)


                attn_tensor = torch.stack(padded_attention_tensors)
                attention_dispersions.append(torch.std(attn_tensor.mean(dim=1)).item())

            # Обработка hidden states
          if output.hidden_states is not None and len(output.hidden_states) > 0:
              last_layer = output.hidden_states[-1]
              if isinstance(last_layer, tuple):
                last_hidden = last_layer[0]
              else:
                last_hidden = last_layer

              last_hidden = last_hidden[0, -1, :]
              hidden_dispersions.append(torch.std(last_hidden).item())

        # Вычисляем средние метрики по всем сэмплам
        attention_disp = np.mean(attention_dispersions) if attention_dispersions else 0
        hidden_disp = np.mean(hidden_dispersions) if hidden_dispersions else 0
        cleaned_answers = []
        for ans in generated_texts:
              cleaned = clean_llm_answers(ans)
              cleaned_answers.append(cleaned[0] if cleaned else "")

        return {
            **self.knowno_metrics(cleaned_answers),
            "attention_dispersion": attention_disp,
            "hidden_state_dispersion": hidden_disp,
            'llm_answers': cleaned_answers
        }

    def knowno_metrics(self, texts) ->dict:

        embeddings = self.sbert.encode(texts)
        similarities = []
        for i in range(len(embeddings)):
            for j in range(i+1, len(embeddings)):
                sim = 1 - cosine(embeddings[i], embeddings[j])
                similarities.append(sim)

        avg_similarity = np.mean(similarities) if similarities else 0

        return {
            "knowno_semantic_similarity": avg_similarity,
            "knowno_confidence": float(avg_similarity > 0.3)
        }

    def whitebox_metrics(self, attentions, hiddens) -> dict:

        attn_means = torch.stack([attn.mean(dim=0) for attn in attentions])
        hiddens_stack = torch.stack(hiddens)

        attention_dispersion = torch.std(attn_means.mean(dim=1)).item()
        hsd = torch.std(hiddens_stack.mean(dim=0)).item()

        return {
            "attention_dispersion": attention_dispersion,
            "hidden_state_dispersion": hsd
        }

    def detect_ambiguity(self, prompt, threshold):
        metrics = self.prompt_analyze(prompt)
        knowno = metrics.get("knowno_semantic_similarity", 0.0)
        attn = metrics.get("attention_dispersion", 0.0)
        hidden = metrics.get("hidden_state_dispersion", 0.0)

        score = 0.3 * (1 - knowno) + 0.4 * attn + 0.3 * hidden
        return int(score > threshold), metrics

detector = AmbiguityDetector(model, tokenizer)
TEST_FILE = "test_results.json"

def convert_numpy_types(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, torch.Tensor):
        return obj.cpu().numpy().tolist()
    raise TypeError(f"Object of type {obj.__class__.__name__} is not serializable")

RESULTS_FILE = "calibration_results.json"
calibration_results = []
if os.path.exists(RESULTS_FILE):
    with open(RESULTS_FILE, "r") as f:
        calibration_results = json.load(f)

start_idx = len(calibration_results)
for idx, row in tqdm(calibration_data.iloc[start_idx:].iterrows(),
                    total=len(calibration_data) - start_idx,
                    desc="Калибровка",
                    initial=start_idx):

  full_question = f"Question:{calibration_data['question']}\nEvironment: {calibration_data['environment_short']}\n"
  score, calibr_analysis = detector.detect_ambiguity(full_question, threshold=0.5)

  calibration_results.append(calibr_analysis)
  if calibration_results:
          try:
              with open(RESULTS_FILE, "w") as f:
                  json.dump(calibration_results, f, default=convert_numpy_types, indent=4)  # Используем indent для удобного форматирования
          except Exception as e:
            print(f"Ошибка при записи в файл: {e}")

def run_testing(detector, test_data, optimal_threshold):
    test_results = []
    start_idx = len(test_results)
    for idx, row in tqdm(test_data.iloc[start_idx:].iterrows(),
                        total=len(test_data)-start_idx,
                        desc="Тестирование",
                        initial=start_idx):
      try:

        full_question = f"Question:{row['question']}\nEvironment: {row['environment_short']}\n"
        score, analysis = detector.detect_ambiguity(full_question, threshold=0.5)

        # Формирование результата
        result = {
            "llm_answers": analysis.get('llm_answers', []),
                "scores": score,
                "y_amb_type": row["ambiguity_type"],
                "y_amb_intents": row["user_intent"],
                "y_amb_shortlist": row["amb_shortlist"],
        }

        test_results.append(result)
      except KeyError as ke:
            print(f"Ошибка при обработке строки {idx}: Отсутствует ключ в данных: {ke}")
            continue
      except Exception as e:
            print(f"Ошибка при обработке строки {idx}: {e}")
            continue

      if test_results:
          try:
              with open(TEST_FILE, "w") as f:
                  json.dump(test_results, f, default=convert_numpy_types, indent=4)  # Используем indent для удобного форматирования
          except Exception as e:
            print(f"Ошибка при записи в файл: {e}")

    return test_results

#Тестирование
test_df = run_testing(detector, test_data, optimal_threshold=0.65)

metrics_batch = batch_metric_calculation(
    llm_answers_batch=[item['llm_answers'] for item in test_df],
    scores=[item['scores'] for item in test_df],
    y_amb_type_batch=[item['y_amb_type'] for item in test_df],
    y_amb_intents_batch=[item['y_amb_intents'] for item in test_df],
    y_amb_shortlist_batch=[item['y_amb_shortlist'] for item in test_df]
)

print(metrics_batch.keys())

print(sum(metrics_batch['scores']))

columns = ["llm_answers", "scores", "y_amb_type", "y_amb_intents", "y_amb_shortlist", "SR", "help_rate", "correct_help_rate", "SSC"]
table = wandb.Table(columns=columns)

batch_size = len(next(iter(metrics_batch.values())))

for i in range(batch_size):
    llm_answer = metrics_batch.get("llm_answers", [])[i] if len(metrics_batch.get("llm_answers", [])) > i else ""
    score = metrics_batch.get("scores", [])[i] if len(metrics_batch.get("scores", [])) > i else 0
    y_amb_type = metrics_batch.get("y_amb_type", [])[i] if len(metrics_batch.get("y_amb_type", [])) > i else ""
    y_amb_intents = metrics_batch.get("y_amb_intents", [])[i] if len(metrics_batch.get("y_amb_intents", [])) > i else ""
    y_amb_shortlist = metrics_batch.get("y_amb_shortlist", [])[i] if len(metrics_batch.get("y_amb_shortlist", [])) > i else ""
    SR = metrics_batch.get("SR", [])[i] if len(metrics_batch.get("SR", [])) > i else 0
    help_rate = metrics_batch.get("help_rate", [])[i] if len(metrics_batch.get("help_rate", [])) > i else 0
    correct_help_rate = metrics_batch.get("correct_help_rate", [])[i] if len(metrics_batch.get("correct_help_rate", [])) > i else 0
    SSC = metrics_batch.get("SSC", [])[i] if len(metrics_batch.get("SSC", [])) > i else 0

    table.add_data(llm_answer, score, y_amb_type, y_amb_intents, y_amb_shortlist, SR, help_rate, correct_help_rate, SSC)

table

wandb.log({"metrics_table": table})

wandb.log({
    "llm_answers": metrics_batch.get("llm_answers", ""),
    "scores": metrics_batch.get("scores", 0),
    "y_amb_type": metrics_batch.get("y_amb_type", ""),
    "y_amb_intents": metrics_batch.get("y_amb_intents", ""),
    "y_amb_shortlist": metrics_batch.get("y_amb_shortlist", ""),
    "SR": metrics_batch.get("SR", 0),
    "help_rate": metrics_batch.get("help_rate", 0),
    "correct_help_rate": metrics_batch.get("correct_help_rate", 0),
    "SSC": metrics_batch.get("SSC", 0),
           })

wandb.finish()


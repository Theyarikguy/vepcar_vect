import pandas as pd
import re
from gensim.models import Word2Vec

# ---------- 1. Загрузка данных ----------
df = pd.read_csv(
    'texts_olo_full.csv',
    encoding='utf-8',
    quotechar='"'
)

texts = df['full_text'].dropna().astype(str).tolist()

# Вот здесь была ошибка: count из itertools — это не length.
print(f'Загружено текстов: {len(texts)}')

# ---------- 2. Токенизация ----------
def tokenize(text: str) -> list[str]:
    text = text.lower()
    tokens = re.findall(r'\b\w+\b', text, flags=re.UNICODE)
    return tokens

sentences = [tokenize(t) for t in texts]

# Небольшая диагностика — полезно видеть, что реально идёт в модель
total_tokens = sum(len(s) for s in sentences)
print(f'Всего токенов: {total_tokens}')
print(f'Пример предложения: {sentences[0][:15]}')

# ---------- 3. Обучение Word2Vec ----------
model = Word2Vec(
    sentences=sentences,
    vector_size=100,
    window=5,
    min_count=5,
    workers=4,
    sg=1,
    epochs=20,
    seed=42
)

# ---------- 4. Сохранение ----------
model.save('word2vec_karelian.model')
model.wv.save('word2vec_karelian.kv')

print(f'Обучено слов: {len(model.wv)}')
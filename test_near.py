from gensim.models import Word2Vec

# ---------- Загрузка модели ----------
# Загружаем полную модель (если сохраняли через model.save)
model = Word2Vec.load('word2vec_karelian.model')
wv = model.wv   # объект KeyedVectors


def normalize(word: str) -> str:
    """Приводим слово к тому же виду, что и при обучении (нижний регистр)."""
    return word.strip().lower()


def similarity(w1: str, w2: str) -> None:
    w1, w2 = normalize(w1), normalize(w2)

    # Проверяем, что оба слова есть в словаре модели
    if w1 not in wv:
        print(f"⚠️  Слово '{w1}' отсутствует в словаре модели.")
        return
    if w2 not in wv:
        print(f"⚠️  Слово '{w2}' отсутствует в словаре модели.")
        return

    score = wv.similarity(w1, w2)          # косинусное сходство, от -1 до 1
    # Переведём в проценты для наглядности
    print(f"Сходство '{w1}' и '{w2}': {score:.4f}  ({score * 100:.1f}%)")


def main() -> None:
    print("Введите два слова через пробел (или 'exit' для выхода).")
    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nВыход.")
            break

        if not line:
            continue
        if line.lower() in {'exit', 'quit', 'q', 'выход'}:
            print("Выход.")
            break

        parts = line.split()
        if len(parts) != 2:
            print("Нужно ровно два слова. Попробуйте ещё раз.")
            continue

        similarity(parts[0], parts[1])


if __name__ == '__main__':
    main()
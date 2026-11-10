from gensim.models import Word2Vec

# ---------- Загрузка модели ----------
model = Word2Vec.load('word2vec_karelian.model')
wv = model.wv   # объект KeyedVectors


def normalize(word: str) -> str:
    """Приводим слово к тому же виду, что и при обучении."""
    return word.strip().lower()


def nearest(word: str, topn: int = 10) -> None:
    w = normalize(word)

    if w not in wv:
        print(f"⚠️  Слово '{w}' отсутствует в словаре модели.")
        # Подскажем похожие по написанию слова (не по смыслу)
        similar_spelling = wv.key_to_index.keys()
        candidates = [k for k in similar_spelling if w[:3] in k][:5]
        if candidates:
            print(f"   Возможно, вы имели в виду: {', '.join(candidates)}")
        return

    print(f"\nБлижайшие соседи для '{w}':")
    print(f"  {'слово':<25}{'сходство':>10}")
    print("  " + "-" * 35)
    for neighbor, score in wv.most_similar(w, topn=topn):
        print(f"  {neighbor:<25}{score:>10.4f}")
    print()


def main() -> None:
    print("Введите слово, чтобы найти его ближайших соседей.")
    print("Можно указать число: '<слово> <N>' (по умолчанию N=10).")
    print("'exit' — выход.\n")

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
        word = parts[0]

        # Необязательный второй аргумент — количество соседей
        topn = 10
        if len(parts) >= 2:
            try:
                topn = int(parts[1])
                if topn <= 0:
                    raise ValueError
            except ValueError:
                print("Количество соседей должно быть положительным целым числом.")
                continue

        nearest(word, topn)


if __name__ == '__main__':
    main()
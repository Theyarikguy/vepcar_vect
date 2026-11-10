import pandas as pd
from itertools import combinations
from gensim.models import Word2Vec

# ============================================================
# Параметры
# ============================================================
MODEL_PATH     = 'word2vec_karelian.model'
MEANINGS_PATH  = 'meanings_olo.csv.zst'
MEMBERS_PATH   = 'synset_members_olo.csv.zst'

TOP_N          = 5     # блок 1: попадает ли слово в топ-N соседей
SIM_THRESHOLD  = 0.7     # блок 2: порог косинусного сходства (>=)
OUT_CSV        = 'synonym_check_results.csv'

# ============================================================
# 1. Загрузка модели и данных
# ============================================================

def count_true(df: pd.DataFrame, column: str) -> tuple[int, int, float]:
    """
    Считает статистику по булевой колонке.

    Возвращает кортеж (n_true, n_total, percent), где
    n_true  — число True,
    n_total — общее число строк,
    percent — доля True в процентах.
    """
    n_total = len(df)
    n_true  = int(df[column].sum())
    percent = 100.0 * n_true / n_total if n_total else 0.0
    return n_true, n_total, percent


def print_block_stats(df: pd.DataFrame,
                      column: str,
                      label: str) -> None:
    """Печатает строку статистики для одного блока проверки."""
    n_true, n_total, percent = count_true(df, column)
    print(f'{label:<38}{n_true:>5d} / {n_total:<5d}  ({percent:5.1f}%)')

print('Загрузка модели и данных...')
model = Word2Vec.load(MODEL_PATH)
wv = model.wv

meanings = pd.read_csv(MEANINGS_PATH, compression='zstd', encoding='utf-8')
members  = pd.read_csv(MEMBERS_PATH,  compression='zstd', encoding='utf-8')

meanings = meanings.dropna(subset=['meaning_id', 'lemma'])

gloss_col = 'meaning_ru' if 'meaning_ru' in meanings.columns else 'lemma'

meaning2lemma = {
    mid: str(l).strip().lower()
    for mid, l in zip(meanings['meaning_id'], meanings['lemma'])
}
meaning2gloss = {
    mid: str(g).strip()
    for mid, g in zip(meanings['meaning_id'], meanings[gloss_col])
}

# synset_id -> { lemma -> (meaning_id, gloss) }
synset_data = {}
for sid, group in members.groupby('synset_id'):
    d = {}
    for mid in group['meaning_id']:
        lemma = meaning2lemma.get(mid)
        if lemma:
            # одна лемма может соответствовать нескольким meaning_id,
            # но для Word2Vec это один и тот же токен — берём первый
            d.setdefault(lemma, (mid, meaning2gloss.get(mid, '')))
    synset_data[sid] = d

# ============================================================
# 2. Сравнение с синонимическими множествами
# ============================================================
results = []
skipped = []          # леммы, которых нет в словаре модели

for sid, lemma_map in synset_data.items():
    lemmas   = list(lemma_map.keys())
    in_vocab = [w for w in lemmas if w in wv]

    for w in lemmas:
        if w not in wv:
            skipped.append({
                'synset_id': sid,
                'lemma'    : w,
                'gloss'    : lemma_map[w][1][:60],
                'reason'   : 'не в словаре модели',
            })

    if len(in_vocab) < 2:
        continue

    for w1, w2 in combinations(sorted(in_vocab), 2):
        sim = float(wv.similarity(w1, w2))

        top1 = {w for w, _ in wv.most_similar(w1, topn=TOP_N)}
        top2 = {w for w, _ in wv.most_similar(w2, topn=TOP_N)}
        in_top_n = (w2 in top1) or (w1 in top2)

        above_threshold = sim >= SIM_THRESHOLD

        results.append({
            'synset_id'      : sid,
            'word1'          : w1,
            'gloss1'         : lemma_map[w1][1][:40],
            'word2'          : w2,
            'gloss2'         : lemma_map[w2][1][:40],
            'similarity'     : round(sim, 4),
            'in_top_n'       : in_top_n,
            'above_threshold': above_threshold,
        })

df      = pd.DataFrame(results)
skipped = pd.DataFrame(skipped)

# ============================================================
# 3. Вывод
# ============================================================
pd.set_option('display.width', 220)
pd.set_option('display.max_colwidth', 42)
pd.set_option('display.max_rows', None)

print('\n=== Параметры ===')
print(f'Блок 1: попадание в топ-{TOP_N} соседей')
print(f'Блок 2: сходство >= {SIM_THRESHOLD}')

if df.empty:
    print('\nНе найдено ни одной пары — проверьте, что модель и CSV-файлы совпадают.')
else:
    n_synsets = df['synset_id'].nunique()
    print(f'\n=== Таблица результатов ({len(df)} пар из {n_synsets} синсетов) ===')
    print(df.to_string(index=False))

    # --- Сводка ---
    n       = len(df)
    n_top   = int(df['in_top_n'].sum())
    n_thr   = int(df['above_threshold'].sum())
    n_both  = int((df['in_top_n'] & df['above_threshold']).sum())
    n_eith  = int((df['in_top_n'] | df['above_threshold']).sum())

    print('\n=== Сводка ===')
    print(f'Блок 1 (в топ-{TOP_N}):                {n_top:4d} / {n}  ({100*n_top/n:5.1f}%)')
    print(f'Блок 2 (сходство >= {SIM_THRESHOLD}):       {n_thr:4d} / {n}  ({100*n_thr/n:5.1f}%)')
    print(f'Оба условия выполнены:             {n_both:4d} / {n}  ({100*n_both/n:5.1f}%)')
    print(f'Хотя бы одно условие:              {n_eith:4d} / {n}  ({100*n_eith/n:5.1f}%)')
    print('\n=== Сводка по блокам ===')
    print(f'{"блок":<38}{"верно":>5} / {"всего":<5}  (доля)')
    print('-' * 62)

    print_block_stats(df, 'in_top_n',
                      f'Блок 1: в топ-{TOP_N} соседей')

    print_block_stats(df, 'above_threshold',
                      f'Блок 2: сходство >= {SIM_THRESHOLD}')

    print_block_stats(df.assign(both=df['in_top_n'] & df['above_threshold']),
                      'both',
                      'Оба блока выполнены одновременно')

    print_block_stats(df.assign(either=df['in_top_n'] | df['above_threshold']),
                      'either',
                      'Хотя бы один блок выполнен')

    # --- Пары, провалившие оба блока (потенциально «плохие» синонимы) ---
    fail_both = (
        df[~df['in_top_n'] & ~df['above_threshold']]
        .sort_values('similarity')
    )
    if not fail_both.empty:
        print(f'\n=== Пары, провалившие оба блока ({len(fail_both)}) ===')
        print(fail_both.to_string(index=False))

    # --- Лучшие пары ---
    print('\n=== Топ-10 пар по сходству ===')
    print(df.sort_values('similarity', ascending=False).head(10).to_string(index=False))

    # --- Сохранение ---
    df.to_csv(OUT_CSV, index=False)
    print(f'\nПолные результаты сохранены в {OUT_CSV}')

    print_block_stats(df, 'in_top_n',
                      f'Блок 1: в топ-{TOP_N} соседей')

    print_block_stats(df, 'above_threshold',
                      f'Блок 2: сходство >= {SIM_THRESHOLD}')

    print_block_stats(df.assign(both=df['in_top_n'] & df['above_threshold']),
                      'both',
                      'Оба блока выполнены одновременно')

    print_block_stats(df.assign(either=df['in_top_n'] | df['above_threshold']),
                      'either',
                      'Хотя бы один блок выполнен')

# --- Диагностика по пропущенным словам ---
# if not skipped.empty:
#     print(f'\n=== Пропущенные леммы ({len(skipped)}) — отсутствуют в словаре модели ===')
#     print(skipped.to_string(index=False))
#     print('\nСовет: увеличьте корпус, уменьшите min_count при обучении '
#           'или перейдите на FastText для редких словоформ.')
#

"""Сквозной конвейер: данные -> признаки -> метки -> CSV -> проверка формата -> счёт на 5 минутах.

    .venv/bin/python run.py baseline            # копия базового решения организаторов (±50)
    .venv/bin/python run.py v0                  # признаки + Витерби
    .venv/bin/python run.py v1                  # бустинг на внешних матчах + Витерби на 4 состояния
    .venv/bin/python run.py score out/*.csv     # оценить готовые файлы
    .venv/bin/python run.py summary             # сводка по трекингу (без сырых значений)

Счёт на 5 минутах — только проверка, что конвейер верный (кадры, формат, смысл меток).
По правилам выбирать версии и подгонять параметры по нему НЕЛЬЗЯ.
Данные матча не показывать облачным ИИ: печатаем только сводные цифры.
"""

import argparse
import filecmp
import os
import sys
import time

from mtah import atd, features, metrics, submission
from mtah.config import OUT_DIR, REF_DIR, PUBLIC_GT
from mtah.methods import baseline, viterbi, boost


def score_files(paths, gt_path=None):
    gt = metrics.codes_from_labels(submission.load_labels(gt_path or PUBLIC_GT))
    print("\nПроверка: {}".format("открытые 5 минут (17 владений; это НЕ рейтинг и НЕ для подбора)"
                                 if not gt_path else gt_path))
    print("{:<28} {:>8} {:>8} {:>6} {:>7}".format("файл", "macroF1", "spellF1", "влад.", "ср.с"))
    for path in paths:
        err = submission.validate(path)
        name = os.path.splitext(os.path.basename(path))[0]
        if err:
            print("{:<28} НЕ ПРОШЁЛ ПРОВЕРКУ: {}".format(name, err))
            continue
        r = metrics.evaluate(gt, metrics.codes_from_labels(submission.load_labels(path)))
        print("{:<28} {:>8.4f} {:>8.4f} {:>6} {:>7.1f}".format(
            name, r["macro_f1"], r["spell_f1"], r["n_pred_spells"], r["pred_mean_dur_s"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["baseline", "v0", "v1", "score", "summary"])
    ap.add_argument("files", nargs="*")
    ap.add_argument("--window", type=int, default=50)
    ap.add_argument("--name", default=None)
    ap.add_argument("--gt", default=None, help="другой файл ответов (например, синтетика)")
    args = ap.parse_args()

    if args.cmd == "score":
        score_files(args.files, args.gt)
        return

    t = time.time()
    d = atd.load()
    print("трекинг прочитан за {:.0f} с".format(time.time() - t))
    if args.cmd == "summary":
        for k, v in atd.summary(d).items():
            print("  {}: {}".format(k, v))
        return

    if args.cmd == "baseline":
        labels = baseline.run(d, window=args.window)
        name = args.name or ("baseline_smooth{}".format(args.window) if args.window else "baseline_raw")
    elif args.cmd == "v1":
        from mtah.methods import boost_train
        models = boost_train.train()
        labels = boost.run(d, models, boost_train.decoder_params())
        name = args.name or "v1"
    else:
        f = features.build(d)
        labels = viterbi.run(d, f)
        name = args.name or "v0"

    path = submission.write(os.path.join(OUT_DIR, name + ".csv"), labels)
    err = submission.validate(path)
    print("файл: {} — {}".format(path, "формат в порядке" if not err else "ОШИБКА: " + err))
    if args.cmd == "baseline":
        ref = os.path.join(REF_DIR, name + ".csv")
        if os.path.isfile(ref):
            same = filecmp.cmp(path, ref, shallow=False)
            print("сверка с файлом организаторов {}: {}".format(
                os.path.basename(ref), "СОВПАДАЕТ байт в байт — чтение и кадры верны" if same
                else "НЕ совпадает — искать ошибку в чтении трекинга/выравнивании"))
    score_files([path], args.gt)
    if err:
        sys.exit(1)


if __name__ == "__main__":
    main()

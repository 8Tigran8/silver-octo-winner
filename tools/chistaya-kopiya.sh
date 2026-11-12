#!/usr/bin/env bash
# Проверка воспроизводимости: чистая копия репозитория (последний коммит) в отдельной папке ->
# перевод внешних матчей из сырых файлов -> run.py v1 -> sha256. Две копии в разных папках должны
# дать один и тот же файл байт в байт, и он должен совпасть с out/v1.csv основной папки.
#
#   tools/chistaya-kopiya.sh [папка1 папка2]
#
# Данные матча и сырые внешние матчи не в git: их подключаем ссылками (только чтение).
# Окружение Python — то же .venv (версии закреплены в requirements.txt).
set -euo pipefail
cd "$(dirname "$0")/.."
SRC=$(pwd)
PY="$SRC/.venv/bin/python"
A=${1:-/tmp/mtah-chistaya-a}
B=${2:-/tmp/mtah-chistaya-b}
for D in "$A" "$B"; do
  rm -rf "$D"; mkdir -p "$D"
  git archive HEAD | tar -x -C "$D"
  ln -s "$SRC/data" "$D/data"
  mkdir -p "$D/external" "$D/ref"
  ln -s "$SRC/external/idsse" "$D/external/idsse"
  ln -s "$SRC/external/metrica" "$D/external/metrica"
  cp "$SRC"/ref/*.csv "$D/ref/"
  echo "== $D: внешние матчи из сырых файлов"
  (cd "$D" && "$PY" tools/ext_idsse.py >/dev/null && "$PY" tools/ext_metrica.py >/dev/null)
  echo "== $D: run.py v1"
  (cd "$D" && "$PY" run.py v1 | grep -E "файл:|обучен|v1 ")
done
echo "== контрольные суммы"
sha256sum "$SRC/out/v1.csv" "$A/out/v1.csv" "$B/out/v1.csv"
n=$(sha256sum "$SRC/out/v1.csv" "$A/out/v1.csv" "$B/out/v1.csv" | awk '{print $1}' | sort -u | wc -l)
[ "$n" -eq 1 ] && echo "ВОСПРОИЗВОДИТСЯ: три файла совпадают байт в байт" || { echo "НЕ совпадает"; exit 1; }

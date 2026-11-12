#!/usr/bin/env bash
# День прихода данных: одна команда от архива организаторов до годного файла сдачи.
#
#   tools/den-dannyh.sh [архив.zip]        # архив (если дан) распакует в data/
#
# Шаги: распаковка -> проверка файлов -> справочные CSV (ref/) -> сводка трекинга (только цифры) ->
# копия базового решения + сверка байт в байт с файлом организаторов -> v0 -> v1 -> проверка формата ->
# счёт на открытых 5 минутах (ТОЛЬКО проверка конвейера; подбирать по нему нельзя) -> копия под именем команды.
#
# Данные матча облачным ИИ не показываем: скрипт печатает только сводные цифры.
# Для прогона на фальшивом матче: MTAH_DATA_DIR=/tmp/mtah-synth MTAH_GT=/tmp/mtah-synth/truth.csv tools/den-dannyh.sh
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
DATA="${MTAH_DATA_DIR:-data}"
TEAM="${MTAH_TEAM:-}"          # точное название команды; пока не задано — файл только out/v0.csv
GT_ARGS=()
[ -n "${MTAH_GT:-}" ] && GT_ARGS=(--gt "$MTAH_GT")

echo "== 1. Данные"
if [ $# -ge 1 ]; then
  mkdir -p "$DATA"
  case "$1" in
    *.zip) unzip -n -q "$1" -d "$DATA" ;;
    *.tar*|*.tgz) tar -xf "$1" -C "$DATA" ;;
    *) echo "не знаю, как распаковать $1"; exit 1 ;;
  esac
  # если архив принёс лишнюю верхнюю папку — поднимаем tactical/ и broadcast/ на уровень data/
  if [ ! -d "$DATA/tactical" ]; then
    inner=$(find "$DATA" -maxdepth 3 -type d -name tactical | head -1)
    [ -n "$inner" ] && mv "$(dirname "$inner")"/* "$DATA"/
  fi
fi
for f in tactical/BAR-ATH_tactical_atd.xml tactical/BAR-ATH_tactical_atd.txt; do
  [ -s "$DATA/$f" ] || { echo "НЕТ $DATA/$f — распакуй архив организаторов в $DATA/"; exit 1; }
done
echo "файлы трекинга на месте"

echo "== 2. Справочные файлы организаторов"
[ -s ref/baseline_smooth50.csv ] && [ -s ref/BAR-ATH_possession_gt_public.csv ] || tools/fetch_ref.sh
$PY -m tests.test_metrics | tail -3

echo "== 3. Сводка трекинга (только цифры)"
$PY run.py summary

echo "== 4. Копия базового решения: должна совпасть байт в байт"
out=$($PY run.py baseline "${GT_ARGS[@]}" | tee /dev/stderr)
if [ -z "${MTAH_GT:-}" ] && ! grep -q "СОВПАДАЕТ байт в байт" <<<"$out"; then
  echo "СТОП: базовое решение не совпало с файлом организаторов — чтение трекинга или кадры не те."
  echo "Файл сдачи v0 всё равно соберу, но верить ему нельзя, пока не найдена причина."
  BAD=1
fi

echo "== 5. Наш v0 (параметры подобраны только на внешних матчах)"
$PY run.py v0 "${GT_ARGS[@]}"

echo "== 6. Наш v1 (бустинг на внешних матчах + Витерби на 4 состояния; ~15 мин)"
BEST=out/v0.csv
if ls external/conv/*.npz >/dev/null 2>&1; then
  $PY run.py v1 "${GT_ARGS[@]}" && BEST=out/v1.csv
else
  echo "нет внешних матчей (external/conv) — v1 пропущен, сдаём v0. Подготовка: tools/ext_idsse.py, tools/ext_metrica.py"
fi

if [ -n "$TEAM" ]; then
  cp "$BEST" "out/${TEAM}.csv"
  $PY -c "from mtah import submission as s; e=s.validate('out/${TEAM}.csv'); print('out/${TEAM}.csv:', e or 'формат в порядке')"
else
  echo "название команды не задано (MTAH_TEAM) — файл сдачи пока $BEST"
fi
echo "контрольные суммы:"; sha256sum out/v0.csv; [ -f out/v1.csv ] && sha256sum out/v1.csv
[ -z "${BAD:-}" ] || exit 2

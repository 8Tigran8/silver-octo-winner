#!/usr/bin/env bash
# Скачивает справочные CSV организаторов в ref/: открытые 5 минут ответов и четыре их базовых решения.
# Только данные (CSV), без кода. Источник: github.com/metrica-sports/more-than-a-hack-2027
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p ref
BASE=https://raw.githubusercontent.com/metrica-sports/more-than-a-hack-2027/main/evaluation
curl -fsSL "$BASE/gt/BAR-ATH_possession_gt_public.csv" -o ref/BAR-ATH_possession_gt_public.csv
for v in raw smooth25 smooth50 smooth125; do
  curl -fsSL "$BASE/results/baseline_$v.csv" -o "ref/baseline_$v.csv"
done
ls -la ref

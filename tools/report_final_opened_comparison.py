"""Render every registered outcome; never changes decision thresholds."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'))
from final_opened_comparison import DOC,OUT,OLD,NEW,check,summary,sha

def main():
 check();result=json.loads((DOC/'results.json').read_text());decision=json.loads((DOC/'decision.json').read_text());p=json.loads((DOC/'plan.json').read_text())
 if summary(result['records'])!=decision:raise ValueError('Saved decision differs from registered computation')
 for source in result['sources']:
  path=OUT/source['fixture']['game_id']/'predictions.npz'
  if sha(path)!=source['predictions_sha256']:raise ValueError('Prediction artifact changed')
 rows=result['records'];idx={(r['game'],r['scenario'],r['candidate'],r['reference']):r for r in rows}
 text=['# Разовое сравнение двух замороженных версий на 18 матчах','',
 f"**Выбор по заранее записанному правилу: `{decision['selected']}`.** Все защитные условия новой версии: {'PASS' if decision['new_passes'] else 'FAIL'}. Это выбор между двумя готовыми моделями, без дообучения, настройки decoder или исправления ответов по разметке.",'',
 'Матчи PFF ЧМ-2022 ранее открывались. Они не входили в обучение весов, но прежние результаты были известны; теперь мы используем их и для решения о релизе. Поэтому это не свежая независимая оценка обобщения. Два варианта восстановления эталона (canonical и SDK-shadow) также не являются двумя независимыми наборами.','',
 '## Средний Spell F1 и регрессии','',
 '| Группа | Сценарий | Старая | Новая | Изменение | Победы | Худший матч Δ | 95% bootstrap Δ | Условия |',
 '|---|---|---:|---:|---:|---:|---:|---|---|']
 for r in decision['comparisons']:
  if r['reference']!='canonical':continue
  lo,hi=r['bootstrap95'];failed=','.join(k for k,v in r['checks'].items() if not v)
  text.append(f"| {r['cohort']} | {r['scenario']} | {r['old_spell']:.6f} | {r['new_spell']:.6f} | {r['gain']:+.6f} | {r['wins']}/{r['games']} | {r['worst_gain']:+.6f} | [{lo:+.6f}, {hi:+.6f}] | {failed or 'PASS'} |")
 text+=['','Условия относятся к каждой группе/сценарию/версии эталона отдельно: средний Spell не снижается, средний macro F1 не снижается более чем на 0.002, худший матч по Spell не теряет более 0.020, средний F1 каждого H/A/D не теряет более 0.010. Дополнительные pooled-проверки обычного/исходного стрессового входа: recall владений <3 с не падает, доли false D, false LIVE и неправильной команды при пропавшем мяче не растут больше чем на 1 процентный пункт.','',
 'Это инженерные правила выбора, а не статистически обоснованные границы неухудшения. Требования прироста +0.02 нет. Bootstrap использует целые парные матчи (10000 выборок, фиксированный seed); общие команды/поставщик и повторное использование данных ограничивают интерпретацию интервала. Проверки тяжёлого шума заранее включены как консервативный запрет принятия: соответствие этого шума конкурсным данным не доказано.','',
 '## Все матчи: исходный трекинг и прежний стресс','',
 '| Матч | Старая natural | Новая natural | Δ | Старая blind | Новая blind | Δ |',
 '|---|---:|---:|---:|---:|---:|---:|']
 for g in p['fixtures']:
  cells=[]
  for s in ['natural_detected','label_blind_seed7']:
   a=idx[g['game_id'],s,OLD,'canonical']['spell_f1_mean'];b=idx[g['game_id'],s,NEW,'canonical']['spell_f1_mean'];cells.extend([f'{a:.6f}',f'{b:.6f}',f'{b-a:+.6f}'])
  text.append('| '+g['home']+'–'+g['away']+' ('+g['game_id']+') | '+' | '.join(cells)+' |')
 text+=['','## Ошибки при пропадании мяча и короткие владения','',
 '| Сценарий | Версия | Recall <3 с | Ложный D среди live | Ложный LIVE среди dead | Неверная команда среди live | <1 с matched/reference |',
 '|---|---|---:|---:|---:|---:|---:|']
 for r in decision['comparisons']:
  if r['cohort']!='all18' or r['reference']!='canonical':continue
  for c in ['old','new']:
   rates=r['rates'][c];counts=r['counts'][c];cells=[f"{rates[k]*100:.2f}%" for k in ['short_recall','false_dead','false_live','wrong_team']]
   text.append('| '+r['scenario']+' | '+c+' | '+' | '.join(cells)+f" | {counts['subsecond_matched']}/{counts['subsecond_reference']} |")
 text+=['','## Полный перечень нарушений','']
 failures=[r for r in decision['comparisons'] if not r['pass_all']]
 if not failures:text.append('Нарушений зарегистрированных условий нет в обеих версиях эталона.')
 for r in failures:text.append('- '+r['cohort']+' / '+r['scenario']+' / '+r['reference']+': '+', '.join(k for k,v in r['checks'].items() if not v))
 text+=['','## Протокол и границы доказательств','',
 'Для каждой игры оба алгоритма получили одинаковую геометрию и независимые границы периодов. Парный расчёт точно совпал с обоими рабочими entrypoint на обучающем матче 3816: исходный вход и один пресет ВВ, без оценки качества. Старые предсказания natural/blind повторили предыдущий аудит побайтно на всех 18 играх. Повторно восстановленные эталоны совпали с прежними массивами. Скрипт сохранял все предсказания матча до открытия его эталона. Защита блокировала доступ к данным и разметке конкурсного матча.','',
 'Пресеты ВВ: 10/15 Гц, независимый координатный шум 0.5/2 м, выпадения мяча 2–6 с. Алгоритм из PR №2 применён к нашей канонической сетке PFF 25 Гц, а не к заново считанным нативным отсчётам датчика. Это важно: нельзя считать такое искажение точной моделью другого поставщика или судейской проверки. Исходные пропуски сохраняются, никакой разметки в генераторе искажений нет. Шесть сценариев остаются 18 матчами, а не 108 независимыми наблюдениями.','',
 'Metrica Game 3 не включён: его уже исследовали, уровень турнира скрыт, новым независимым матчем считать нельзя. Дополнительных данных не скачивали. Реальные улучшения на Ла Лиге и отсутствие переобучения этим аудитом не доказаны.','',
 'Отдельное сообщение ВВ подтверждает два совпавших запуска принятой до аудита версии на его Linux ARM сервере. Это сообщение коллеги, а не проверенный в этой итерации машинный журнал; детали в vv_external_reproduction.json.','',
 'План: plan.json. Хеши исходников/моделей/данных: freeze.json. Все 432 записи метрик: results.json. Выбор и все защитные проверки: decision.json. Raw tracking, разметка и per-frame predictions остаются локальными. Организаторам ничего не отправлялось; публичность репозитория не менялась.']
 (DOC/'REPORT.md').write_text('\n'.join(text)+'\n');print('Report written:',DOC/'REPORT.md')
if __name__=='__main__':main()

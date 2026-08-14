"""Самодостаточная страница демонстрационного помощника."""

MARKETPLACE_UI = """<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Помощник аналитика маркетплейса</title>
  <style>
    :root { color-scheme: dark; font-family: Inter, system-ui, sans-serif; }
    * { box-sizing: border-box; }
    body { margin: 0; background: #0b1020; color: #edf1ff; }
    main { max-width: 980px; margin: auto; padding: 48px 20px; }
    h1 { font-size: clamp(30px, 5vw, 54px); margin: 0 0 8px; }
    .lead { color: #aab5d6; font-size: 18px; margin-bottom: 32px; }
    .panel { background: #151c32; border: 1px solid #293451; border-radius: 18px; padding: 22px; margin-bottom: 20px; }
    label { display: block; color: #aab5d6; margin: 14px 0 7px; }
    input, button { width: 100%; border-radius: 10px; padding: 13px; font: inherit; }
    input { color: #edf1ff; background: #0d1427; border: 1px solid #34415f; }
    button { margin-top: 18px; border: 0; background: #7767ff; color: white; font-weight: 700; cursor: pointer; }
    button:disabled { opacity: .6; cursor: wait; }
    .grid { display: grid; grid-template-columns: repeat(auto-fit,minmax(210px,1fr)); gap: 12px; }
    .metric { padding: 16px; border-radius: 14px; background: #0d1427; }
    .metric strong { display: block; font-size: 25px; color: #8de1bd; margin-top: 5px; }
    .warn strong { color: #ffb57a; }
    h2 { margin-top: 0; }
    li { margin: 8px 0; color: #d4daf0; }
    .error { color: #ff9292; }
    .hidden { display: none; }
    footer { color: #74809f; font-size: 13px; }
  </style>
</head>
<body><main>
  <h1>AI-помощник аналитика</h1>
  <p class="lead">Загрузите обезличенный CSV и спросите о выкупе или возвратах.</p>
  <section class="panel">
    <label for="report">Текущий CSV-отчёт</label>
    <input id="report" type="file" accept=".csv,text/csv">
    <label for="previous-report">Предыдущий CSV — необязательно, только для сравнения</label>
    <input id="previous-report" type="file" accept=".csv,text/csv">
    <label for="question">Вопрос</label>
    <input id="question" value="Почему процент выкупа маленький?">
    <label for="threshold">Порог низкого выкупа, %</label>
    <input id="threshold" type="number" min="0" max="100" value="70">
    <label for="return-threshold">Порог высокой доли возвратов, %</label>
    <input id="return-threshold" type="number" min="0" max="100" value="15">
    <button id="ask">Проанализировать</button>
    <p id="error" class="error hidden"></p>
  </section>
  <section id="result" class="hidden">
    <div class="panel"><h2>Объяснение AI</h2><p id="explanation"></p><p><small id="knowledge"></small></p></div>
    <div class="panel"><h2>Вывод</h2><p id="answer"></p></div>
    <div class="panel"><h2>Показатели</h2><div id="metrics" class="grid"></div></div>
    <div class="panel"><h2>Что известно точно</h2><ul id="facts"></ul></div>
    <div class="panel"><h2>Возможные причины</h2><ul id="causes"></ul></div>
    <div class="panel"><h2>Каких данных не хватает</h2><ul id="missing"></ul></div>
  </section>
  <footer>Демонстрационные данные не отправляются в облачный AI API. Не загружайте закрытые данные без разрешения владельца.</footer>
<script>
const $ = id => document.getElementById(id);
const fill = (id, items) => $(id).innerHTML = items.map(x => `<li>${escapeHtml(x)}</li>`).join('');
const escapeHtml = value => { const d=document.createElement('div'); d.textContent=value; return d.innerHTML; };
$('ask').onclick = async () => {
  const file = $('report').files[0];
  $('error').classList.add('hidden'); $('result').classList.add('hidden');
  if (!file) { $('error').textContent='Выберите CSV-файл.'; $('error').classList.remove('hidden'); return; }
  $('ask').disabled=true; $('ask').textContent='Считаю…';
  try {
    const previous=$('previous-report').files[0];
    const comparing=Boolean(previous);
    const url=comparing?'/v1/marketplace/compare-chat-upload':'/v1/marketplace/chat-upload';
    const body=comparing?{question:$('question').value, previous_filename:previous.name, previous_csv_text:await previous.text(), current_filename:file.name, current_csv_text:await file.text()}:{question:$('question').value, filename:file.name, csv_text:await file.text(), low_threshold:Number($('threshold').value), high_return_threshold:Number($('return-threshold').value)};
    const response = await fetch(url, {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(body)});
    const payload=await response.json(); if(!response.ok) throw new Error(payload.detail || 'Ошибка анализа');
    const data=comparing?payload.comparison:payload.analysis;
    $('explanation').textContent=payload.explanation;
    $('knowledge').textContent='Справочник: '+payload.knowledge_sources.join(', ');
    $('answer').textContent=data.answer; fill('facts',data.facts); fill('causes',data.possible_causes); fill('missing',data.missing_data);
    $('metrics').innerHTML=data.metrics.map(m=>{ if(data.analysis_type==='comparison'){const warn=m.change_pp<0; return `<div class="metric ${warn?'warn':''}"><span>${escapeHtml(m.product)}</span><strong>${m.change_pp>0?'+':''}${m.change_pp} п.п.</strong><small>${m.previous_buyout_rate}% → ${m.current_buyout_rate}%</small></div>`;} const returns=data.analysis_type==='returns'; const rate=returns?m.return_rate:m.buyout_rate; const warn=returns?rate>Number($('return-threshold').value):rate<Number($('threshold').value); const detail=returns?`${m.returned} из ${m.bought} выкупленных`:`${m.bought} из ${m.ordered}`; return `<div class="metric ${warn?'warn':''}"><span>${escapeHtml(m.product)}</span><strong>${rate}%</strong><small>${detail}</small></div>`; }).join('');
    $('result').classList.remove('hidden');
  } catch(e) { $('error').textContent=e.message; $('error').classList.remove('hidden'); }
  finally { $('ask').disabled=false; $('ask').textContent='Проанализировать'; }
};
</script></main></body></html>"""

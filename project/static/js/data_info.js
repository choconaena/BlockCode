const dataSelect = document.getElementById('data-select');
const infoType  = document.getElementById('info-type');
const paramN    = document.getElementById('param-n');
const infoDiv   = document.getElementById('data-info');

infoType.addEventListener('change', () => {
  paramN.hidden = !['sample','images'].includes(infoType.value);
  loadInfo();
});
dataSelect.addEventListener('change', loadInfo);

function loadInfo() {
  const file = dataSelect.value;
  const type = infoType.value;
  const n    = paramN.hidden ? 0 : paramN.value;
  fetch(`/data-info?file=${file}&type=${type}&n=${n}`)
    .then(r => r.json())
    .then(info => {
      let html = '';
      if (type === 'shape') {
        html = `<p>행: ${info.rows}, 열: ${info.cols}</p>`;
      } else if (type === 'structure') {
        html = '<ul>';
        info.columns.forEach(c => html += `<li>${c.name}: ${c.dtype}</li>`);
        html += '</ul>';
      } else if (type === 'sample') {
        html = '<table><tr>' + info.columns.map(c=>`<th>${c}</th>`).join('') + '</tr>';
        info.sample.forEach(rw => html += `<tr>${rw.map(v=>`<td>${v}</td>`).join('')}</tr>`);
        html += '</table>';
      } else if (type === 'images') {
        html = '<div style="display:flex;flex-wrap:wrap;">';
        info.images.forEach(b64 => html += `<img src="data:image/png;base64,${b64}">`);
        html += '</div>';
      }
      infoDiv.innerHTML = html;
    });
}

// 초기 호출
infoType.dispatchEvent(new Event('change'));

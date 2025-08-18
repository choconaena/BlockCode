// static/js/logs.js
(function(){
  const logView = document.getElementById('log-view');
  let evtSrc = null;

  function startStream(stage){
    if (!logView) return;
    if (evtSrc) evtSrc.close();
    logView.textContent = `--- ${stage} 실행 로그 ---\n`;
    evtSrc = new EventSource(`/logs/stream?stage=${encodeURIComponent(stage)}`);
    evtSrc.onmessage = (e)=>{
      logView.textContent += e.data + "\n";
      // 자동 스크롤 연동 (main_log.html에서 이 이벤트를 듣고 있음)
      document.dispatchEvent(new CustomEvent('aib-log-line'));
    };
    evtSrc.onerror = ()=>{
      logView.textContent += "[SSE] 연결 종료 또는 오류\n";
      evtSrc && evtSrc.close();
      document.dispatchEvent(new CustomEvent('aib-log-line'));
    };
  }

  function run(stage){
    fetch(`/run/${stage}`, {method: 'POST'})
      .then(r => r.json())
      .then(res => {
        if (res.ok) {
          document.getElementById('tab-log')?.click();   // 로그 탭으로 전환
          startStream(stage);
        } else {
          alert(`실행 실패: ${res.error || 'unknown'}`);
        }
      })
      .catch(err => alert(`실행 오류: ${err}`));
  }

  // 페이지 내 모든 실행 버튼 바인딩 (코드 패널 & 로그 탭)
  document.querySelectorAll('[data-run]').forEach(btn=>{
    btn.addEventListener('click', ()=> run(btn.dataset.run));
  });
})();
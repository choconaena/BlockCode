// static/js/tabs.js
(function(){
  // 메인 탭 토글: code/data/log
  function showMainTab(key){
    ['code','data','log'].forEach(k=>{
      document.getElementById('tab-'+k)?.classList.toggle('active', k===key);
      document.getElementById('content-'+k)?.classList.toggle('active', k===key);
    });
  }

  document.getElementById('tab-code')?.addEventListener('click', ()=>showMainTab('code'));
  document.getElementById('tab-data')?.addEventListener('click', ()=>showMainTab('data'));
  document.getElementById('tab-log') ?.addEventListener('click', ()=>showMainTab('log'));

  // 스테이지 탭 토글: pre/model/train/eval
  const stageBtns = document.querySelectorAll('.stage-tab');
  function showStage(stage){
    stageBtns.forEach(b=>b.classList.toggle('active', b.dataset.stage===stage));
    document.querySelectorAll('.stage-pane').forEach(p=>{
      p.hidden = (p.id !== ('pane-code-' + stage));
    });
  }
  stageBtns.forEach(btn=>{
    btn.addEventListener('click', ()=> showStage(btn.dataset.stage));
  });

  // 초기 활성화 (로컬스토리지 복원은 layout.html 내부 스크립트가 처리)
  if (!document.querySelector('.stage-tab.active') && stageBtns[0]){
    stageBtns[0].classList.add('active');
    showStage(stageBtns[0].dataset.stage);
  }
})();
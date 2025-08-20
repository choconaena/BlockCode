// static/js/logs.js
(function(){
  const logView = document.getElementById('log-view');
  let evtSrc = null;

  // 🔹 사용자 ID 가져오는 함수
  function getCurrentUserId() {
    const userIdInput = document.getElementById('user-id-input');
    return userIdInput ? userIdInput.value.trim() || 'anonymous' : 'anonymous';
  }

  function startStream(stage){
    if (!logView) return;
    if (evtSrc) evtSrc.close();
    
    const userId = getCurrentUserId();
    logView.textContent = `--- ${stage} 실행 로그 (사용자: ${userId}) ---\n`;
    
    // 🔹 사용자 ID를 쿼리 파라미터로 포함
    evtSrc = new EventSource(`/logs/stream?stage=${encodeURIComponent(stage)}&user_id=${encodeURIComponent(userId)}`);
    
    evtSrc.onmessage = (e)=>{
      logView.textContent += e.data + "\n";
      // 자동 스크롤 연동
      document.dispatchEvent(new CustomEvent('aib-log-line'));
    };
    evtSrc.onerror = ()=>{
      logView.textContent += "[SSE] 연결 종료 또는 오류\n";
      evtSrc && evtSrc.close();
      document.dispatchEvent(new CustomEvent('aib-log-line'));
    };
  }

  function run(stage){
    const userId = getCurrentUserId();
    
    // 🔹 사용자 ID 검증
    if (!userId || userId === 'anonymous') {
      alert('사용자 ID를 입력해주세요!');
      const userIdInput = document.getElementById('user-id-input');
      if (userIdInput) userIdInput.focus();
      return;
    }
    
    console.log(`실행 요청: stage=${stage}, user_id=${userId}`); // 디버그 로그
    
    // 🔹 방법 1: FormData 방식
    const formData = new FormData();
    formData.append('user_id', userId);
    
    // 🔹 방법 2: JSON 방식 (백업)
    const jsonData = {
      user_id: userId
    };
    
    // FormData 방식 시도
    fetch(`/run/${stage}`, {
      method: 'POST',
      body: formData
    })
      .then(r => {
        console.log('응답 상태:', r.status);
        return r.json();
      })
      .then(res => {
        console.log('응답 데이터:', res);
        if (res.ok) {
          document.getElementById('tab-log')?.click();   // 로그 탭으로 전환
          startStream(stage);
        } else {
          // FormData 실패 시 JSON 방식 재시도
          if (res.error && res.error.includes('user_id가 필요합니다')) {
            console.log('FormData 실패, JSON 방식으로 재시도...');
            return fetch(`/run/${stage}`, {
              method: 'POST',
              headers: {
                'Content-Type': 'application/json'
              },
              body: JSON.stringify(jsonData)
            })
            .then(r => r.json())
            .then(res2 => {
              if (res2.ok) {
                document.getElementById('tab-log')?.click();
                startStream(stage);
              } else {
                alert(`실행 실패: ${res2.error || 'unknown'}`);
              }
            });
          } else {
            alert(`실행 실패: ${res.error || 'unknown'}`);
          }
        }
      })
      .catch(err => {
        console.error('실행 요청 오류:', err);
        alert(`실행 오류: ${err}`);
      });
  }

  // 페이지 내 모든 실행 버튼 바인딩 (코드 패널 & 로그 탭)
  document.querySelectorAll('[data-run]').forEach(btn=>{
    btn.addEventListener('click', ()=> run(btn.dataset.run));
  });
})();
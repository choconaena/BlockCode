// static/js/sidebar.js

// ---- 공통: 블록 활성/비활성 토글 ----
function setBlockActive(block, active) {
  block.dataset.active = active ? 'true' : 'false';
  const isRequired = block.classList.contains('block-required');

  // 필수 블록은 항상 enable, 선택 블록만 토글
  block.querySelectorAll('input, select, textarea, button').forEach(el => {
    el.disabled = isRequired ? false : !active;
    
    if (!active && !isRequired) {
      el.removeAttribute('required');
      el.removeAttribute('min');
      el.removeAttribute('max');
    }
  });
}

// ---- 좌측 대분류(전처리/모델/학습/평가) 탭 전환 ----
function initLeftTabs() {
  document.querySelectorAll('.left-tab').forEach(tab => {
    tab.addEventListener('click', () => {
      document.querySelectorAll('.left-tab').forEach(t => t.classList.remove('active'));
      tab.classList.add('active');
      const key = tab.dataset.tab; // pre, model, train, eval
      document.querySelectorAll('.pane').forEach(p => p.hidden = true);
      document.getElementById('pane-' + key).hidden = false;
    });
  });
  // 초기 기본 탭
  const first = document.querySelector('.left-tab[data-tab="pre"]');
  if (first) first.click();
}

// ---- 블록 초기화 (data-active / block-required 기준) ----
function initBlocks() {
  document.querySelectorAll('.block').forEach(block => {
    const isRequired = block.classList.contains('block-required');
    const wantActive = (block.dataset.active || 'false') === 'true';
    setBlockActive(block, isRequired || wantActive);
  });

  // 블록 박스 클릭 시 토글 (내부 컨트롤 클릭은 무시, 필수 블록은 토글 불가)
  document.querySelectorAll('.block').forEach(block => {
    block.addEventListener('click', e => {
      const tag = e.target.tagName;
      if (['INPUT','SELECT','LABEL','OPTION','TEXTAREA','BUTTON'].includes(tag)) return;
      if (block.classList.contains('block-required')) return;
      const now = block.dataset.active === 'true';
      setBlockActive(block, !now);
    });
  });
}

// ---- 의존 필드 (테스트 여부, 라벨 필터) ----
function initDependentFields() {
  const isTest = document.getElementById('is_test');
  if (isTest) {
    const syncTest = () => {
      const yes = isTest.value === 'true';
      const testDiv  = document.getElementById('test-div');
      const ratioDiv = document.getElementById('ratio-div');
      if (testDiv)  testDiv.style.display  = yes ? 'block' : 'none';
      if (ratioDiv) ratioDiv.style.display = yes ? 'none' : 'block';
    };
    isTest.addEventListener('change', syncTest);
    syncTest();
  }

  const dropBad = document.getElementById('drop_bad');
  if (dropBad) {
    const syncBad = () => {
      const p = document.getElementById('drop_bad_params');
      if (p) p.style.display = dropBad.checked ? 'block' : 'none';
    };
    dropBad.addEventListener('change', syncBad);
    syncBad();
  }
}

// ---- 🔹 AJAX 폼 제출 처리 (사용자 ID 포함) ----
function initAjaxFormSubmit() {
  const form = document.querySelector('form[method="post"]');
  if (!form) return;

  form.addEventListener('submit', async (e) => {
    e.preventDefault(); // 기본 폼 제출 방지
    
    try {
      // 사용자 ID 검증
      const userIdInput = document.getElementById('user-id-input');
      const userId = userIdInput.value.trim();
      
      if (!userId || userId === 'anonymous') {
        showNotification('사용자 ID를 입력해주세요!', 'error');
        userIdInput.focus();
        return;
      }
      
      // 제출 버튼에서 stage 값 가져오기
      const clickedButton = document.activeElement;
      const stage = clickedButton.getAttribute('value') || 'all';
      
      // 로딩 상태 표시
      const originalText = clickedButton.textContent;
      clickedButton.textContent = '변환 중...';
      clickedButton.disabled = true;
      
      // 폼 데이터 준비 (기존 로직 유지)
      document.querySelectorAll('.block[data-active="false"]').forEach(block => {
        if (!block.classList.contains('block-required')) {
          block.querySelectorAll('input, select, textarea').forEach(el => {
            el.removeAttribute('name');
            el.removeAttribute('required');
            el.removeAttribute('min');
            el.removeAttribute('max');
          });
        }
      });
      
      document.querySelectorAll('.block[data-active="true"]').forEach(block => {
        block.querySelectorAll('input, select, textarea, button').forEach(el => {
          el.disabled = false;
        });
      });
      
      document.querySelectorAll('.block-required').forEach(block => {
        block.querySelectorAll('input, select, textarea, button').forEach(el => el.disabled = false);
      });

      // FormData 생성 및 stage, user_id 추가
      const formData = new FormData(form);
      formData.set('stage', stage);
      formData.set('user_id', userId);  // 🔹 사용자 ID 포함
      
      console.log('전송할 데이터:', {
        stage: stage,
        user_id: userId,
        // 기타 폼 데이터들...
      });
      
      // AJAX 요청
      const response = await fetch('/convert', {
        method: 'POST',
        body: formData
      });
      
      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`HTTP ${response.status}: ${errorText}`);
      }
      
      const codeText = await response.text();
      
      // 코드 탭으로 전환하고 결과 표시
      document.getElementById('tab-code')?.click();
      
      if (stage === 'all') {
        // 전체 변환: 모든 코드 패널에 각각 표시
        updateAllCodePanels(codeText);
      } else {
        // 특정 스테이지: 해당 패널만 업데이트
        updateSingleCodePanel(stage, codeText);
        // 해당 스테이지 탭으로 전환
        document.querySelector(`.stage-tab[data-stage="${stage}"]`)?.click();
      }
      
      // 성공 메시지
      showNotification(`코드 변환 완료! (사용자: ${userId})`, 'success');
      
    } catch (error) {
      console.error('폼 제출 오류:', error);
      showNotification(`오류 발생: ${error.message}`, 'error');
    } finally {
      // 버튼 상태 복원
      const clickedButton = document.activeElement;
      clickedButton.textContent = originalText;
      clickedButton.disabled = false;
    }
  });
}

// ---- 코드 패널 업데이트 함수들 ----
function updateAllCodePanels(combinedCode) {
  // 구분자로 분리
  const sections = combinedCode.split('# ==========');
  const stageMap = {
    '전처리': 'pre',
    '모델 설계': 'model', 
    '학습': 'train',
    '평가': 'eval'
  };
  
  sections.forEach(section => {
    if (section.trim()) {
      for (const [korName, engStage] of Object.entries(stageMap)) {
        if (section.includes(korName)) {
          const code = section.split('==========')[1]?.trim() || '';
          updateCodeDisplay(engStage, code);
          break;
        }
      }
    }
  });
}

function updateSingleCodePanel(stage, code) {
  updateCodeDisplay(stage, code);
}

function updateCodeDisplay(stage, code) {
  const pane = document.getElementById(`pane-code-${stage}`);
  if (pane) {
    const codeElement = pane.querySelector('.code');
    if (codeElement) {
      codeElement.textContent = code;
    }
  }
}

// ---- 알림 표시 함수 ----
function showNotification(message, type = 'info') {
  // 기존 알림 제거
  const existing = document.querySelector('.notification');
  if (existing) existing.remove();
  
  const notification = document.createElement('div');
  notification.className = `notification notification-${type}`;
  notification.textContent = message;
  notification.style.cssText = `
    position: fixed; top: 20px; right: 20px; z-index: 1000;
    padding: 12px 20px; border-radius: 4px; color: white;
    background: ${type === 'success' ? '#4CAF50' : type === 'error' ? '#f44336' : '#2196F3'};
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    animation: slideIn 0.3s ease-out;
  `;
  
  document.body.appendChild(notification);
  
  // 3초 후 자동 제거
  setTimeout(() => {
    notification.style.animation = 'slideOut 0.3s ease-in';
    setTimeout(() => notification.remove(), 300);
  }, 3000);
}

// ---- 초기화 ----
document.addEventListener('DOMContentLoaded', () => {
  initLeftTabs();
  initBlocks();
  initDependentFields();
  initAjaxFormSubmit(); // 🔹 새로 추가
});

// ---- CSS 애니메이션 추가 ----
const style = document.createElement('style');
style.textContent = `
  @keyframes slideIn {
    from { transform: translateX(100%); opacity: 0; }
    to { transform: translateX(0); opacity: 1; }
  }
  @keyframes slideOut {
    from { transform: translateX(0); opacity: 1; }
    to { transform: translateX(100%); opacity: 0; }
  }
`;
document.head.appendChild(style);
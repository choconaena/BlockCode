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
// 탭 전환은 "숨기기/보이기"만! 절대 disabled를 건드리지 않습니다.
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
      // 여기서 disabled는 건드리지 않습니다 (전송 직전 일괄 처리)
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

// ---- 폼 전송 직전: 모든 "활성 블록" 입력을 enable 강제 ----
function initFormSubmitEnableAll() {
  const form = document.querySelector('form[method="post"]');
  if (!form) return;
  form.addEventListener('submit', () => {
    // 비활성화된 블록의 validation 속성 제거 (에러 방지)
    document.querySelectorAll('.block[data-active="false"]').forEach(block => {
      if (!block.classList.contains('block-required')) {
        block.querySelectorAll('input, select, textarea').forEach(el => {
          el.removeAttribute('name');  // name 제거로 서버 전송 방지
          el.removeAttribute('required');
          el.removeAttribute('min');
          el.removeAttribute('max');
        });
      }
    });
    
    // 활성화된 모든 블록에서 disabled 풀기
    document.querySelectorAll('.block[data-active="true"]').forEach(block => {
      block.querySelectorAll('input, select, textarea, button').forEach(el => {
        el.disabled = false;
      });
    });
    // 필수 블록들(dataset 등)은 이미 enable 상태이지만, 한 번 더 보장
    document.querySelectorAll('.block-required').forEach(block => {
      block.querySelectorAll('input, select, textarea, button').forEach(el => el.disabled = false);
    });
  });
}

document.addEventListener('DOMContentLoaded', () => {
  initLeftTabs();
  initBlocks();
  initDependentFields();
  initFormSubmitEnableAll();
});
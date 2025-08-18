# generators/evaluation.py
# 평가 코드 생성기

from .base import BaseGenerator


class EvaluationGenerator(BaseGenerator):
    """평가 코드 생성기"""
    
    def generate(self, form_data):
        """평가 코드 생성"""
        self.clear_blocks()
        
        # 파라미터 추출
        params = self._extract_parameters(form_data)
        
        # 코드 생성
        self._add_imports()
        self._add_configuration(params)
        self._add_data_loading()
        self._add_model_loading()
        self._add_evaluation_functions(params)
        self._add_visualization(params)
        self._add_main_block(params)
        
        return self.render_blocks()
    
    def _extract_parameters(self, form_data):
        """폼 데이터에서 파라미터 추출"""
        # 메트릭 리스트 추출
        metrics = self.get_form_list(form_data, 'metrics')
        
        return {
            # 메트릭
            'metrics': metrics,
            'average': self.get_form_value(form_data, 'average', 'macro'),
            'topk_k': int(self.get_form_value(form_data, 'topk_k', 3)),
            
            # 리포트
            'show_classification_report': self.is_checked(form_data, 'show_classification_report'),
            'show_confusion_matrix': self.is_checked(form_data, 'show_confusion_matrix'),
            'cm_normalize': self.is_checked(form_data, 'cm_normalize'),
            
            # 시각화
            'viz_samples': int(self.get_form_value(form_data, 'viz_samples', 0)),
            'viz_mis': int(self.get_form_value(form_data, 'viz_mis', 0)),
            
            # 설정
            'eval_batch': int(self.get_form_value(form_data, 'eval_batch', 128)),
            'num_classes': int(self.get_form_value(form_data, 'num_classes', 10)),
            'class_names': self.get_form_value(form_data, 'class_names', ''),
            'force_cpu': self.is_checked(form_data, 'force_cpu')
        }
    
    def _add_imports(self):
        """임포트 문 추가"""
        code = '''# 자동 생성된 evaluation.py
# AI 블록코딩 - 평가 파이프라인

import os
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import json
import datetime

# 로깅 유틸
def log(msg):
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[eval][{timestamp}] {msg}")

log("=== EVALUATION START ===")
'''
        self.add_block("imports", code, required=True)
    
    def _add_configuration(self, params):
        """설정 추가"""
        # 클래스 이름 처리
        if params['class_names']:
            class_names_code = f"[s.strip() for s in '''{params['class_names']}'''.split(',') if s.strip()]"
        else:
            class_names_code = f"[str(i) for i in range({params['num_classes']})]"
        
        code = f'''
# 평가 설정
CONFIG = {{
    'batch_size': {params['eval_batch']},
    'num_classes': {params['num_classes']},
    'force_cpu': {params['force_cpu']},
    'metrics': {params['metrics']},
    'average': '{params['average']}',
    'topk_k': {params['topk_k']}
}}

# 클래스 이름
class_names = {class_names_code}
if len(class_names) != CONFIG['num_classes']:
    log(f"Warning: class_names length ({{len(class_names)}}) != num_classes ({{CONFIG['num_classes']}})")
    class_names = [str(i) for i in range(CONFIG['num_classes'])]

# 디바이스 설정
if CONFIG['force_cpu']:
    device = torch.device('cpu')
else:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
log(f"Using device: {{device}}")
'''
        self.add_block("configuration", code, required=True)
    
    def _add_data_loading(self):
        """데이터 로딩 추가"""
        code = '''
# 데이터 로드
def load_data():
    """전처리된 데이터 로드"""
    WORKDIR = os.environ.get("AIB_WORKDIR", ".")
    data_path = os.path.join(WORKDIR, "data", "dataset.pt")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}. Run preprocessing first.")
    
    data = torch.load(data_path, map_location='cpu')
    
    X_test = data['X_test']
    y_test = data['y_test']
    
    log(f"Test data loaded: {len(X_test)} samples")
    
    # DataLoader 생성
    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(
        test_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        num_workers=0
    )
    
    return test_loader, X_test, y_test
'''
        self.add_block("data_loading", code, required=True)
    
    def _add_model_loading(self):
        """모델 로딩 추가"""
        code = '''
# 모델 로드
def load_model():
    """학습된 모델 로드"""
    from model import build_model, load_model
    
    WORKDIR = os.environ.get("AIB_WORKDIR", ".")
    checkpoint_path = os.path.join(WORKDIR, "artifacts", "best_model.pth")
    
    # 모델 생성 및 로드
    model = build_model()
    model = load_model(model, checkpoint_path, map_location=device)
    model = model.to(device)
    model.eval()
    
    return model
'''
        self.add_block("model_loading", code, required=True)
    
    def _add_evaluation_functions(self, params):
        """평가 함수 추가"""
        code = '''
# 추론 실행
def run_inference(model, loader):
    """모델 추론 실행"""
    all_preds = []
    all_labels = []
    all_probs = []
    
    model.eval()
    with torch.no_grad():
        for data, target in tqdm(loader, desc="Evaluating"):
            data = data.to(device)
            
            # 추론
            output = model(data)
            probs = torch.softmax(output, dim=1)
            preds = output.argmax(dim=1)
            
            # 결과 저장
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(target.numpy())
            all_probs.extend(probs.cpu().numpy())
    
    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


# 메트릭 계산
def calculate_metrics(y_true, y_pred, y_proba=None):
    """평가 메트릭 계산"""
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        classification_report, confusion_matrix
    )
    
    results = {}
    '''
        
        # 메트릭별 코드 추가
        if 'accuracy' in params['metrics']:
            code += '''
    # Accuracy
    if 'accuracy' in CONFIG['metrics']:
        results['accuracy'] = accuracy_score(y_true, y_pred)
        log(f"Accuracy: {results['accuracy']:.4f}")
    '''
        
        if 'precision' in params['metrics']:
            code += f'''
    # Precision
    if 'precision' in CONFIG['metrics']:
        results['precision'] = precision_score(y_true, y_pred, average='{params['average']}', zero_division=0)
        log(f"Precision ({params['average']}): {{results['precision']:.4f}}")
    '''
        
        if 'recall' in params['metrics']:
            code += f'''
    # Recall
    if 'recall' in CONFIG['metrics']:
        results['recall'] = recall_score(y_true, y_pred, average='{params['average']}', zero_division=0)
        log(f"Recall ({params['average']}): {{results['recall']:.4f}}")
    '''
        
        if 'f1' in params['metrics']:
            code += f'''
    # F1-score
    if 'f1' in CONFIG['metrics']:
        results['f1'] = f1_score(y_true, y_pred, average='{params['average']}', zero_division=0)
        log(f"F1-score ({params['average']}): {{results['f1']:.4f}}")
    '''
        
        if 'topk' in params['metrics']:
            code += f'''
    # Top-K Accuracy
    if 'topk' in CONFIG['metrics'] and y_proba is not None:
        k = CONFIG['topk_k']
        topk_correct = 0
        for i in range(len(y_true)):
            topk_preds = np.argpartition(-y_proba[i], k-1)[:k]
            if y_true[i] in topk_preds:
                topk_correct += 1
        results['topk_accuracy'] = topk_correct / len(y_true)
        log(f"Top-{{k}} Accuracy: {{results['topk_accuracy']:.4f}}")
    '''
        
        if 'auc' in params['metrics']:
            code += '''
    # AUC (이진 분류)
    if 'auc' in CONFIG['metrics'] and y_proba is not None:
        from sklearn.metrics import roc_auc_score
        try:
            if CONFIG['num_classes'] == 2:
                results['auc'] = roc_auc_score(y_true, y_proba[:, 1])
                log(f"AUC: {results['auc']:.4f}")
            else:
                log("AUC is only supported for binary classification")
        except Exception as e:
            log(f"AUC calculation failed: {e}")
    '''
        
        # Classification Report
        if params['show_classification_report']:
            code += '''
    # Classification Report
    log("\\nClassification Report:")
    report = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)
    print(report)
    results['classification_report'] = report
    '''
        
        # Confusion Matrix
        if params['show_confusion_matrix']:
            code += '''
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    results['confusion_matrix'] = cm.tolist()
    log("\\nConfusion Matrix:")
    print(cm)
    '''
        
        code += '''
    return results
'''
        self.add_block("evaluation_functions", code, required=True)
    
    def _add_visualization(self, params):
        """시각화 함수 추가"""
        if not (params['viz_samples'] > 0 or params['viz_mis'] > 0 or params['show_confusion_matrix']):
            return
        
        code = '''
# 시각화 함수
def visualize_results(y_true, y_pred, X_test, results):
    """결과 시각화"""
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # GUI 없는 환경 지원
    
    WORKDIR = os.environ.get("AIB_WORKDIR", ".")
    artifacts_dir = os.path.join(WORKDIR, "artifacts")
    os.makedirs(artifacts_dir, exist_ok=True)
    '''
        
        # Confusion Matrix 히트맵
        if params['show_confusion_matrix']:
            normalize_str = "True" if params['cm_normalize'] else "False"
            code += f'''
    # Confusion Matrix 히트맵
    if 'confusion_matrix' in results:
        cm = np.array(results['confusion_matrix'])
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # 정규화 옵션
        if {normalize_str}:
            cm_display = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
        else:
            cm_display = cm
            fmt = 'd'
        
        # 히트맵 그리기
        im = ax.imshow(cm_display, interpolation='nearest', cmap='Blues')
        ax.figure.colorbar(im, ax=ax)
        
        # 라벨 설정
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               xticklabels=class_names[:cm.shape[1]],
               yticklabels=class_names[:cm.shape[0]],
               xlabel='Predicted',
               ylabel='True',
               title='Confusion Matrix' + (' (Normalized)' if {normalize_str} else ''))
        
        # 라벨 회전
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # 각 셀에 값 표시
        thresh = cm_display.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm_display[i, j], fmt),
                       ha="center", va="center",
                       color="white" if cm_display[i, j] > thresh else "black")
        
        plt.tight_layout()
        cm_path = os.path.join(artifacts_dir, "confusion_matrix.png")
        plt.savefig(cm_path, dpi=100, bbox_inches='tight')
        plt.close()
        log(f"Confusion matrix saved to {{cm_path}}")
    '''
        
        # 예측 샘플 시각화
        if params['viz_samples'] > 0:
            code += f'''
    # 예측 샘플 시각화
    n_samples = min({params['viz_samples']}, len(X_test))
    if n_samples > 0:
        fig = plt.figure(figsize=(15, 3 * ((n_samples - 1) // 5 + 1)))
        
        for i in range(n_samples):
            ax = fig.add_subplot((n_samples - 1) // 5 + 1, 5, i + 1)
            
            # 이미지 표시
            img = X_test[i].cpu().numpy()
            if img.ndim == 3:
                if img.shape[0] == 1:
                    img = img[0]
                elif img.shape[0] == 3:
                    img = np.transpose(img, (1, 2, 0))
            
            ax.imshow(img, cmap='gray' if img.ndim == 2 else None)
            ax.set_title(f'True: {{class_names[y_true[i]]}}\\nPred: {{class_names[y_pred[i]]}}',
                        fontsize=9)
            ax.axis('off')
        
        plt.tight_layout()
        samples_path = os.path.join(artifacts_dir, "prediction_samples.png")
        plt.savefig(samples_path, dpi=100, bbox_inches='tight')
        plt.close()
        log(f"Prediction samples saved to {{samples_path}}")
    '''
        
        # 오분류 샘플 시각화
        if params['viz_mis'] > 0:
            code += f'''
    # 오분류 샘플 시각화
    misclassified_idx = np.where(y_true != y_pred)[0]
    n_mis = min({params['viz_mis']}, len(misclassified_idx))
    
    if n_mis > 0:
        fig = plt.figure(figsize=(15, 3 * ((n_mis - 1) // 5 + 1)))
        
        for i in range(n_mis):
            idx = misclassified_idx[i]
            ax = fig.add_subplot((n_mis - 1) // 5 + 1, 5, i + 1)
            
            # 이미지 표시
            img = X_test[idx].cpu().numpy()
            if img.ndim == 3:
                if img.shape[0] == 1:
                    img = img[0]
                elif img.shape[0] == 3:
                    img = np.transpose(img, (1, 2, 0))
            
            ax.imshow(img, cmap='gray' if img.ndim == 2 else None)
            ax.set_title(f'True: {{class_names[y_true[idx]]}}\\nPred: {{class_names[y_pred[idx]]}}',
                        fontsize=9, color='red')
            ax.axis('off')
        
        plt.tight_layout()
        mis_path = os.path.join(artifacts_dir, "misclassified_samples.png")
        plt.savefig(mis_path, dpi=100, bbox_inches='tight')
        plt.close()
        log(f"Misclassified samples saved to {{mis_path}}")
    else:
        log("No misclassified samples found!")
'''
        
        self.add_block("visualization", code)
    
    def _add_main_block(self, params):
        """메인 실행 블록 추가"""
        viz_call = ""
        if params['viz_samples'] > 0 or params['viz_mis'] > 0 or params['show_confusion_matrix']:
            viz_call = "\n        # 시각화\n        visualize_results(y_true, y_pred, X_test, results)"
        
        code = f'''
# 메인 실행
if __name__ == "__main__":
    try:
        # 데이터 로드
        test_loader, X_test, y_test = load_data()
        
        # 모델 로드
        model = load_model()
        
        # 추론 실행
        log("Running inference...")
        y_true, y_pred, y_proba = run_inference(model, test_loader)
        
        # 메트릭 계산
        log("\\nCalculating metrics...")
        results = calculate_metrics(y_true, y_pred, y_proba)
        {viz_call}
        
        # 결과 저장
        WORKDIR = os.environ.get("AIB_WORKDIR", ".")
        results_path = os.path.join(WORKDIR, "artifacts", "evaluation_results.json")
        
        # numpy 배열을 리스트로 변환 (JSON 직렬화를 위해)
        save_results = {{}}
        for key, value in results.items():
            if isinstance(value, np.ndarray):
                save_results[key] = value.tolist()
            elif isinstance(value, (int, float, str, list, dict)):
                save_results[key] = value
        
        with open(results_path, 'w') as f:
            json.dump(save_results, f, indent=2)
        
        log(f"\\nResults saved to {{results_path}}")
        log("=== EVALUATION DONE ===")
        
    except Exception as e:
        log(f"Error during evaluation: {{e}}")
        raise
'''
        self.add_block("main", code, required=True)
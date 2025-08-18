# generators/training.py
# 학습 코드 생성기

from .base import BaseGenerator


class TrainingGenerator(BaseGenerator):
    """학습 코드 생성기"""
    
    def generate(self, form_data):
        """학습 코드 생성"""
        self.clear_blocks()
        
        # 파라미터 추출
        params = self._extract_parameters(form_data)
        
        # 코드 생성
        self._add_imports()
        self._add_configuration(params)
        self._add_data_loading()
        self._add_model_setup(params)
        self._add_training_loop(params)
        self._add_main_block()
        
        return self.render_blocks()
    
    def _extract_parameters(self, form_data):
        """폼 데이터에서 파라미터 추출"""
        return {
            # 손실함수
            'loss_method': self.get_form_value(form_data, 'loss_method', 'CrossEntropy'),
            
            # 옵티마이저
            'optimizer_method': self.get_form_value(form_data, 'optimizer_method', 'Adam'),
            'learning_rate': float(self.get_form_value(form_data, 'learning_rate', 0.001)),
            'weight_decay': float(self.get_form_value(form_data, 'weight_decay', 0)),
            'momentum': float(self.get_form_value(form_data, 'momentum', 0.9)),
            
            # 학습 옵션
            'epochs': int(self.get_form_value(form_data, 'epochs', 10)),
            'batch_size': int(self.get_form_value(form_data, 'batch_size', 64)),
            'patience': int(self.get_form_value(form_data, 'patience', 3)),
            'shuffle': self.get_form_value(form_data, 'shuffle', 'true') == 'true',
            'seed': int(self.get_form_value(form_data, 'seed', 42)),
            
            # 스케줄러 (선택)
            'sched_type': self.get_form_value(form_data, 'sched_type', ''),
            'sched_step': int(self.get_form_value(form_data, 'sched_step', 5)),
            'sched_gamma': float(self.get_form_value(form_data, 'sched_gamma', 0.5))
        }
    
    def _add_imports(self):
        """임포트 문 추가"""
        code = '''# 자동 생성된 training.py
# AI 블록코딩 - 학습 파이프라인

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import time
import datetime

# 로깅 유틸
def log(msg):
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[train][{timestamp}] {msg}")
'''
        self.add_block("imports", code, required=True)
    
    def _add_configuration(self, params):
        """설정 추가"""
        code = f'''
# 학습 설정
CONFIG = {{
    'epochs': {params['epochs']},
    'batch_size': {params['batch_size']},
    'learning_rate': {params['learning_rate']},
    'weight_decay': {params['weight_decay']},
    'momentum': {params['momentum']},
    'patience': {params['patience']},
    'shuffle': {params['shuffle']},
    'seed': {params['seed']},
    'loss_method': '{params['loss_method']}',
    'optimizer_method': '{params['optimizer_method']}'
}}

# 시드 설정
torch.manual_seed(CONFIG['seed'])
if torch.cuda.is_available():
    torch.cuda.manual_seed(CONFIG['seed'])

# 디바이스 설정
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
    
    X_train = data['X_train']
    y_train = data['y_train']
    X_test = data['X_test']
    y_test = data['y_test']
    
    log(f"Data loaded: train={len(X_train)}, test={len(X_test)}")
    
    # DataLoader 생성
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=CONFIG['shuffle'],
        num_workers=0
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        num_workers=0
    )
    
    return train_loader, test_loader
'''
        self.add_block("data_loading", code, required=True)
    
    def _add_model_setup(self, params):
        """모델 설정 추가"""
        # 손실함수 매핑
        loss_map = {
            'CrossEntropy': 'nn.CrossEntropyLoss()',
            'MSE': 'nn.MSELoss()'
        }
        
        # 옵티마이저 설정
        if params['optimizer_method'] == 'Adam':
            optimizer_code = f"optim.Adam(model.parameters(), lr={params['learning_rate']}, weight_decay={params['weight_decay']})"
        elif params['optimizer_method'] == 'SGD':
            optimizer_code = f"optim.SGD(model.parameters(), lr={params['learning_rate']}, momentum={params['momentum']}, weight_decay={params['weight_decay']})"
        else:  # RMSprop
            optimizer_code = f"optim.RMSprop(model.parameters(), lr={params['learning_rate']}, weight_decay={params['weight_decay']})"
        
        code = f'''
# 모델, 손실함수, 옵티마이저 설정
def setup_model():
    """모델 및 학습 컴포넌트 설정"""
    from model import build_model
    
    # 모델 생성
    model = build_model().to(device)
    
    # 손실함수
    criterion = {loss_map[params['loss_method']]}
    
    # 옵티마이저
    optimizer = {optimizer_code}
    '''
        
        # 스케줄러 추가 (선택)
        if params['sched_type'] == 'step':
            code += f'''
    # 학습률 스케줄러
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size={params['sched_step']}, gamma={params['sched_gamma']})
    '''
        elif params['sched_type'] == 'cosine':
            code += f'''
    # 학습률 스케줄러
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max={params['epochs']})
    '''
        else:
            code += '''
    scheduler = None
    '''
        
        code += '''
    return model, criterion, optimizer, scheduler
'''
        self.add_block("model_setup", code, required=True)
    
    def _add_training_loop(self, params):
        """학습 루프 추가"""
        code = f'''
# 학습 함수
def train_epoch(model, loader, criterion, optimizer):
    """한 에폭 학습"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    progress_bar = tqdm(loader, desc="Training", ncols=100)
    for batch_idx, (data, target) in enumerate(progress_bar):
        data, target = data.to(device), target.to(device)
        
        # 순전파
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        
        # 역전파
        loss.backward()
        optimizer.step()
        
        # 통계
        total_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)
        
        # 진행 상황 업데이트
        progress_bar.set_postfix({{
            'loss': f'{{loss.item():.4f}}',
            'acc': f'{{100.*correct/total:.2f}}%'
        }})
    
    avg_loss = total_loss / len(loader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy


# 검증 함수
def validate(model, loader, criterion):
    """모델 검증"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        progress_bar = tqdm(loader, desc="Validation", ncols=100)
        for data, target in progress_bar:
            data, target = data.to(device), target.to(device)
            
            output = model(data)
            loss = criterion(output, target)
            
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
    
    avg_loss = total_loss / len(loader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy


# 전체 학습 프로세스
def train_model(model, train_loader, test_loader, criterion, optimizer, scheduler=None):
    """전체 학습 프로세스"""
    WORKDIR = os.environ.get("AIB_WORKDIR", ".")
    checkpoint_dir = os.path.join(WORKDIR, "artifacts")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    best_val_loss = float('inf')
    patience_counter = 0
    history = {{'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}}
    
    log("Starting training...")
    
    for epoch in range(1, CONFIG['epochs'] + 1):
        log(f"\\nEpoch {{epoch}}/{{CONFIG['epochs']}}")
        
        # 학습
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
        
        # 검증
        val_loss, val_acc = validate(model, test_loader, criterion)
        
        # 스케줄러 업데이트
        if scheduler:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            log(f"Learning rate: {{current_lr:.6f}}")
        
        # 결과 출력
        log(f"Train Loss: {{train_loss:.4f}}, Train Acc: {{train_acc:.2f}}%")
        log(f"Val Loss: {{val_loss:.4f}}, Val Acc: {{val_acc:.2f}}%")
        
        # 히스토리 저장
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # 체크포인트 저장
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            checkpoint_path = os.path.join(checkpoint_dir, "best_model.pth")
            torch.save(model.state_dict(), checkpoint_path)
            log(f"✓ Best model saved (val_loss: {{val_loss:.4f}})")
        else:
            patience_counter += 1
            log(f"No improvement ({{patience_counter}}/{{CONFIG['patience']}})")
        
        # Early stopping
        if patience_counter >= CONFIG['patience']:
            log("Early stopping triggered!")
            break
    
    # 최종 체크포인트 저장
    final_checkpoint = os.path.join(checkpoint_dir, "final_model.pth")
    torch.save(model.state_dict(), final_checkpoint)
    
    # 학습 히스토리 저장
    import json
    history_path = os.path.join(checkpoint_dir, "training_history.json")
    with open(history_path, 'w') as f:
        json.dump(history, f)
    
    log(f"Training completed! Best val_loss: {{best_val_loss:.4f}}")
    log(f"Checkpoints saved to {{checkpoint_dir}}")
    
    return history
'''
        self.add_block("training_loop", code, required=True)
    
    def _add_main_block(self):
        """메인 실행 블록 추가"""
        code = '''
if __name__ == "__main__":
    log("=== TRAINING START ===")
    
    try:
        # 데이터 로드
        train_loader, test_loader = load_data()
        
        # 모델 설정
        model, criterion, optimizer, scheduler = setup_model()
        
        # 학습 실행
        history = train_model(
            model, 
            train_loader, 
            test_loader, 
            criterion, 
            optimizer, 
            scheduler
        )
        
        log("=== TRAINING DONE ===")
        
    except Exception as e:
        log(f"Error during training: {e}")
        raise
'''
        self.add_block("main", code, required=True)
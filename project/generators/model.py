# generators/model.py
# 모델 코드 생성기

from .base import BaseGenerator


class ModelGenerator(BaseGenerator):
    """CNN 모델 코드 생성기"""
    
    def generate(self, form_data):
        """모델 코드 생성"""
        self.clear_blocks()
        
        # 파라미터 추출
        params = self._extract_parameters(form_data)
        
        # 코드 생성
        self._add_imports()
        self._add_model_class(params)
        self._add_helper_functions()
        
        return self.render_blocks()
    
    def _extract_parameters(self, form_data):
        """폼 데이터에서 파라미터 추출"""
        params = {
            # 입력층
            'input_w': int(self.get_form_value(form_data, 'input_w', 28)),
            'input_h': int(self.get_form_value(form_data, 'input_h', 28)),
            'input_c': int(self.get_form_value(form_data, 'input_c', 1)),
            
            # Conv1
            'conv1_filters': int(self.get_form_value(form_data, 'conv1_filters', 32)),
            'conv1_kernel': int(self.get_form_value(form_data, 'conv1_kernel', 3)),
            'conv1_stride': int(self.get_form_value(form_data, 'conv1_stride', 1)),
            'conv1_padding': self.get_form_value(form_data, 'conv1_padding', 'same'),
            'conv1_activation': self.get_form_value(form_data, 'conv1_activation', 'relu'),
            
            # Pool1
            'pool1_type': self.get_form_value(form_data, 'pool1_type', 'max'),
            'pool1_size': int(self.get_form_value(form_data, 'pool1_size', 2)),
            'pool1_stride': int(self.get_form_value(form_data, 'pool1_stride', 2)),
            
            # Conv2 (선택)
            'use_conv2': self.is_checked(form_data, 'use_conv2'),
            'conv2_filters': int(self.get_form_value(form_data, 'conv2_filters', 64)),
            'conv2_kernel': int(self.get_form_value(form_data, 'conv2_kernel', 3)),
            'conv2_activation': self.get_form_value(form_data, 'conv2_activation', 'relu'),
            
            # Dropout (선택)
            'use_dropout': self.is_checked(form_data, 'use_dropout'),
            'dropout_p': float(self.get_form_value(form_data, 'dropout_p', 0.25)),
            
            # Dense
            'dense_units': int(self.get_form_value(form_data, 'dense_units', 128)),
            'dense_activation': self.get_form_value(form_data, 'dense_activation', 'relu'),
            
            # Output
            'num_classes': int(self.get_form_value(form_data, 'num_classes', 10)),
            'output_activation': self.get_form_value(form_data, 'output_activation', 'softmax')
        }
        
        # 리사이즈 크기 (전처리에서 온 값 우선)
        resize_n = self.get_form_value(form_data, 'resize_n')
        if resize_n:
            params['input_size'] = int(resize_n)
        else:
            params['input_size'] = min(params['input_w'], params['input_h'])
        
        # 출력 크기 계산
        params['fc_input_size'] = self._calculate_fc_input_size(params)
        
        return params
    
    def _calculate_fc_input_size(self, params):
        """FC 레이어 입력 크기 계산"""
        size = params['input_size']
        channels = params['input_c']
        
        # Conv1 적용
        if params['conv1_padding'] == 'same':
            padding = params['conv1_kernel'] // 2
        else:
            padding = 0
        
        size = (size + 2 * padding - params['conv1_kernel']) // params['conv1_stride'] + 1
        channels = params['conv1_filters']
        
        # Pool1 적용
        if params['pool1_type']:
            size = (size - params['pool1_size']) // params['pool1_stride'] + 1
        
        # Conv2 적용 (선택)
        if params['use_conv2']:
            # Conv2는 same padding 가정
            channels = params['conv2_filters']
        
        return channels * size * size
    
    def _add_imports(self):
        """임포트 문 추가"""
        code = '''# 자동 생성된 model.py
# AI 블록코딩 - CNN 모델 정의

import torch
import torch.nn as nn
import torch.nn.functional as F
'''
        self.add_block("imports", code, required=True)
    
    def _add_model_class(self, params):
        """모델 클래스 추가"""
        # 활성화 함수 매핑
        activation_map = {
            'relu': 'nn.ReLU()',
            'tanh': 'nn.Tanh()',
            'sigmoid': 'nn.Sigmoid()',
            'none': None
        }
        
        # Conv 블록 생성
        conv_layers = []
        
        # Conv1
        if params['conv1_padding'] == 'same':
            padding = params['conv1_kernel'] // 2
        else:
            padding = 0
        
        conv_layers.append(f"            nn.Conv2d({params['input_c']}, {params['conv1_filters']}, "
                          f"kernel_size={params['conv1_kernel']}, padding={padding}),")
        
        if params['conv1_activation'] != 'none':
            conv_layers.append(f"            {activation_map[params['conv1_activation']]},")
        
        # Pool1
        if params['pool1_type'] == 'max':
            conv_layers.append(f"            nn.MaxPool2d({params['pool1_size']}, {params['pool1_stride']}),")
        elif params['pool1_type'] == 'avg':
            conv_layers.append(f"            nn.AvgPool2d({params['pool1_size']}, {params['pool1_stride']}),")
        
        # Conv2 (선택)
        if params['use_conv2']:
            padding2 = params['conv2_kernel'] // 2  # same padding
            conv_layers.append(f"            nn.Conv2d({params['conv1_filters']}, {params['conv2_filters']}, "
                             f"kernel_size={params['conv2_kernel']}, padding={padding2}),")
            
            if params['conv2_activation'] != 'none':
                conv_layers.append(f"            {activation_map[params['conv2_activation']]},")
        
        # Dropout (선택)
        if params['use_dropout']:
            conv_layers.append(f"            nn.Dropout2d(p={params['dropout_p']}),")
        
        # FC 블록 생성
        fc_layers = []
        fc_layers.append(f"            nn.Flatten(),")
        fc_layers.append(f"            nn.Linear({params['fc_input_size']}, {params['dense_units']}),")
        
        if params['dense_activation'] != 'none':
            fc_layers.append(f"            {activation_map[params['dense_activation']]},")
        
        fc_layers.append(f"            nn.Linear({params['dense_units']}, {params['num_classes']}),")
        
        # 출력 활성화 (CrossEntropy 사용 시 불필요)
        if params['output_activation'] == 'softmax':
            fc_layers.append(f"            # Note: Softmax는 일반적으로 loss 함수에서 처리됨")
        
        # 모델 클래스 코드
        code = f'''
class CNN(nn.Module):
    """CNN 모델 클래스"""
    
    def __init__(self):
        super(CNN, self).__init__()
        
        # 합성곱 블록
        self.conv_layers = nn.Sequential(
{chr(10).join(conv_layers)}
        )
        
        # 완전연결 블록
        self.fc_layers = nn.Sequential(
{chr(10).join(fc_layers)}
        )
    
    def forward(self, x):
        """순전파"""
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x
    
    def get_num_parameters(self):
        """파라미터 수 계산"""
        return sum(p.numel() for p in self.parameters())
'''
        self.add_block("model_class", code, required=True)
    
    def _add_helper_functions(self):
        """헬퍼 함수 추가"""
        code = '''
def build_model():
    """모델 생성 함수"""
    model = CNN()
    print(f"Model created with {model.get_num_parameters():,} parameters")
    return model


def load_model(model, checkpoint_path, map_location='cpu'):
    """모델 로드 함수"""
    import os
    if os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location=map_location)
        model.load_state_dict(state_dict)
        print(f"Model loaded from {checkpoint_path}")
    else:
        print(f"Checkpoint not found at {checkpoint_path}")
    return model


def save_model(model, checkpoint_path):
    """모델 저장 함수"""
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Model saved to {checkpoint_path}")


if __name__ == "__main__":
    # 테스트용 코드
    model = build_model()
    
    # 더미 입력으로 테스트
    dummy_input = torch.randn(1, 1, 28, 28)
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    
    # 모델 구조 출력
    print("\\nModel structure:")
    print(model)
'''
        self.add_block("helper_functions", code, required=True)
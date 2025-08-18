# generators/base.py
# 코드 생성기 베이스 클래스

from abc import ABC, abstractmethod
from typing import Dict, Any, List


class CodeBlock:
    """코드 블록 단위"""
    
    def __init__(self, name: str, code: str, required: bool = False):
        self.name = name
        self.code = code
        self.required = required
    
    def render(self) -> str:
        """코드 블록 렌더링"""
        return self.code


class BaseGenerator(ABC):
    """코드 생성기 베이스 클래스"""
    
    def __init__(self):
        self.blocks: List[CodeBlock] = []
    
    @abstractmethod
    def generate(self, form_data: Dict[str, Any]) -> str:
        """코드 생성 (구현 필요)"""
        pass
    
    def add_block(self, name: str, code: str, required: bool = False):
        """코드 블록 추가"""
        self.blocks.append(CodeBlock(name, code, required))
    
    def render_blocks(self) -> str:
        """모든 블록 렌더링"""
        return "\n".join(block.render() for block in self.blocks)
    
    def clear_blocks(self):
        """블록 초기화"""
        self.blocks = []
    
    @staticmethod
    def get_form_value(form_data: Dict, key: str, default: Any = None) -> Any:
        """폼 데이터에서 값 가져오기"""
        return form_data.get(key, default)
    
    @staticmethod
    def get_form_list(form_data: Dict, key: str) -> List[str]:
        """폼 데이터에서 리스트 값 가져오기"""
        value = form_data.get(key, [])
        if isinstance(value, list):
            return value
        return [value] if value else []
    
    @staticmethod
    def is_checked(form_data: Dict, key: str) -> bool:
        """체크박스 체크 여부"""
        value = form_data.get(key)
        return value in ['on', 'true', True, '1']
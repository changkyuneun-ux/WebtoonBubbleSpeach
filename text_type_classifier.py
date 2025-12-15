# -*- coding: utf-8 -*-
import re
from typing import Dict, List, Tuple, Optional

class TextTypeClassifier:
    """
    웹툰 텍스트 유형 분류기 (SFX 필터링 강화판)
    - 기존 인터페이스 호환 유지
    - 딕셔너리 데이터 보강
    """
    
    def __init__(self):
        # 1. 의성어/의태어 (소리/모양) - 이미지 기반 보강
        self.sfx_dictionary = {
            # 충격/폭발
            '쾅', '콰광', '펑', '빵', '뻥', '꽝', '탕', '팡', '퍽', '툭', '턱',
            '쿵', '꿍', '둥', '탁', '딱', '착', '짝', '뚝', '똑', '톡',
            '와장창', '우당탕', '우르르', '와르르', '쨍그랑', '찌그덕',
            
            # 이동/속도
            '뚜벅', '뚜벅뚜벅', '타박타박', '터덜터덜', 
            '덜컹', '덜컹덜컹', '끼익', '삐걱', '스르륵', '슥',
            '획', '휙', '쓱', '훅', '푹', '팍', '쑥', '쌩', '슉',
            '철컥', '찰칵', '딸깍', '달그락', '덜그럭',
            '다그닥', '다그닥다그닥', '따그닥', 
            '질끈', '움찔', '멈칫', '꾸욱', '꾹',
            
            # 자연/동물
            '휘잉', '휭', '쏴', '주르륵', '콸콸', '멍멍', '야옹', 
            
            # 기계/전자/기타
            '부릉', '빵빵', '지지직', '치지직', '반짝', '번쩍',
            '악', '앗', '헉', '헐', '쳇', '흥', '칫', '크윽', '으윽',
            '짜잔', '두둥', '빠밤', '?', '!', '?!', '...', '♪', '♬', '♩'
        }

        # 2. 정규식 패턴
        self.sfx_patterns = [
            r'^(.{1})\1{1,}$',      # 쿵쿵, 쾅쾅 (2회 이상 반복)
            r'^(.{1,2})\1{1,}$',    # 다그닥다그닥 (2글자 반복)
            r'^[ㄱ-ㅎ]+$',           # ㅋㅋㅋ, ㅎㅎㅎ
            r'^[!?~.]+$',           # 기호만 있는 경우
            r'[♪♬♩]',               # 음표 포함
        ]
        
        # 3. 대화체 판단 패턴 (조사/어미)
        self.dialogue_hints = [
             r'[은는이가을를에의로서와과랑하고]', # 조사
             r'(다|요|까|죠|네|게|지|야|해|봐|서|면|고|니|나|군|걸)[\s!?~.]*$' # 어미
        ]
        
        self.sfx_regex = [re.compile(p) for p in self.sfx_patterns]
        self.dialogue_regex = [re.compile(p) for p in self.dialogue_hints]
        
        # 메타 정보 패턴 (기존 코드 호환)
        self.meta_patterns = [
            r'(스토리|작화|글|그림)[\s·:]+', r'제\d+화', r'Episode', r'\d+화',
            r'배달|주문|배송', r'\d+원'
        ]
        self.meta_regex = [re.compile(p, re.IGNORECASE) for p in self.meta_patterns]

    def _normalize(self, text: str) -> str:
        text = text.strip()
        # 문장 끝 특수문자 제거 후 검사를 위해 정제
        return re.sub(r'[!?~.\s]', '', text)

    def is_sfx(self, text: str) -> bool:
        """효과음 여부 강력 체크"""
        if not text: return False
        clean_text = self._normalize(text)
        
        # 1. 사전 매칭
        if clean_text in self.sfx_dictionary: return True
        # 2. 포함 관계 (너무 긴 문장은 제외)
        for sfx in self.sfx_dictionary:
            if len(sfx) > 1 and sfx in clean_text and len(clean_text) < len(sfx) * 3:
                return True
        # 3. 패턴 매칭
        for regex in self.sfx_regex:
            if regex.search(clean_text): return True
        return False

    def is_dialogue(self, text: str) -> bool:
        """대화체 여부 체크"""
        if not text: return False
        for regex in self.dialogue_regex:
            if regex.search(text): return True
        return False

    def classify(self, text: str, is_in_bubble: bool = True) -> Dict:
        """
        텍스트 분류 메인 (기존 반환 형식 유지)
        """
        result = {
            'is_dialogue': False,
            'type': 'unknown',
            'confidence': 0.0,
            'reason': 'unknown',
            'original': text
        }
        
        if not text or not text.strip():
            result['type'] = 'empty'
            result['confidence'] = 1.0
            return result

        clean_text = self._normalize(text)

        # 1. 메타 정보 체크
        for regex in self.meta_regex:
            if regex.search(text):
                result['type'] = 'meta'
                result['reason'] = 'meta_info'
                result['confidence'] = 0.9
                return result

        # 2. 강력한 SFX 체크
        if self.is_sfx(text):
            result['type'] = 'sfx'
            result['reason'] = 'sfx_dictionary'
            result['confidence'] = 0.95
            return result

        # 3. 대화체 판단
        if is_in_bubble:
            # 말풍선 안이라면, SFX가 아닌 이상 대화로 간주 (관대함)
            result['is_dialogue'] = True
            result['type'] = 'dialogue'
            result['confidence'] = 0.8
            result['reason'] = 'bubble_default'
        else:
            # 말풍선 밖(Floating)이라면, 명확한 구조나 웃음소리만 대화로 인정 (엄격함)
            if 'ㅋㅋ' in clean_text or 'ㅎㅎ' in clean_text:
                result['is_dialogue'] = True
                result['type'] = 'dialogue' # UI 호환성을 위해 dialogue로 통일
                result['reason'] = 'floating_laugh'
            elif self.is_dialogue(text) and len(clean_text) > 1:
                result['is_dialogue'] = True
                result['type'] = 'dialogue'
                result['reason'] = 'floating_structure'
            else:
                result['type'] = 'background_noise'
                result['confidence'] = 0.6
                result['reason'] = 'floating_noise'

        return result
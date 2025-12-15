# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
import re
from typing import List, Dict, Optional, Tuple

# 텍스트 분류기 import
try:
    from text_type_classifier import TextTypeClassifier
except ImportError:
    TextTypeClassifier = None

class WebtoonBubbleProcessor:
    """
    웹툰 말풍선 감지 및 분류 통합 처리기 (Optimized v4.1)
    - YOLO 감지 우선 + OCR Padding 적용
    - Floating Text (배경 대사) 감지 로직 통합
    - 기존 Pipeline 코드와 100% 인터페이스 호환
    """
    
    DEFAULT_MODEL_NAME = "yolov8m_seg_speech_bubble.pt"
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        confidence_threshold: float = 0.15,
        use_heuristic: bool = True,   # 호환성 유지용 (실제론 사용 안 함)
        use_ocr: bool = True,
        use_text_filter: bool = True,
        ocr_lang: str = 'kor+eng'
    ):
        self.conf_threshold = confidence_threshold
        self.use_ocr = use_ocr
        self.use_text_filter = use_text_filter
        self.ocr_lang = ocr_lang
        
        # YOLO 모델 로드
        self.yolo_model = None
        self.yolo_available = False
        self._load_yolo_model(model_path)
        
        # 텍스트 분류기
        self.classifier = None
        if use_text_filter and TextTypeClassifier:
            self.classifier = TextTypeClassifier()
            print("✓ 텍스트 분류기 초기화 완료 (Enhanced)")
            
        # OCR 체크
        self.ocr_available = False
        if use_ocr:
            try:
                import pytesseract
                pytesseract.get_tesseract_version()
                self.ocr_available = True
            except:
                print("⚠️ Tesseract 미설치")

    def _load_yolo_model(self, model_path: Optional[str]):
        try:
            from ultralytics import YOLO
            # 모델 경로 찾기 (기존 로직 유지)
            target = model_path
            if not target or not os.path.exists(target):
                search_paths = [
                    os.path.join(os.path.dirname(__file__), self.DEFAULT_MODEL_NAME),
                    os.path.join(os.getcwd(), self.DEFAULT_MODEL_NAME),
                    os.path.join(os.path.dirname(__file__), "models", self.DEFAULT_MODEL_NAME),
                    os.path.join(os.getcwd(), "models", self.DEFAULT_MODEL_NAME),
                ]
                for p in search_paths:
                    if os.path.exists(p):
                        target = p
                        break
            
            if target and os.path.exists(target):
                self.yolo_model = YOLO(target)
                self.yolo_available = True
                print(f"✓ YOLO 모델 로드: {target}")
            else:
                print("⚠️ YOLO 모델을 찾을 수 없습니다.")
        except Exception as e:
            print(f"⚠️ YOLO 로드 실패: {e}")

    def _imread_unicode(self, path: str) -> Optional[np.ndarray]:
        try:
            stream = open(path, "rb")
            bytes = bytearray(stream.read())
            numpyarray = np.asarray(bytes, dtype=np.uint8)
            return cv2.imdecode(numpyarray, cv2.IMREAD_UNCHANGED)
        except:
            return None

    def _preprocess_image(self, img: np.ndarray) -> np.ndarray:
        if len(img.shape) == 2:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:
            return cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        elif img.shape[2] == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def _detect_bubbles_yolo(self, img_rgb: np.ndarray) -> List[Dict]:
        """YOLO 감지"""
        if not self.yolo_available: return []
        
        results = self.yolo_model(img_rgb, verbose=False, conf=self.conf_threshold)
        bubbles = []
        if results:
            boxes = results[0].boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                bubbles.append({
                    'bbox': (x1, y1, x2, y2),
                    'confidence': conf
                })
        bubbles.sort(key=lambda b: b['bbox'][1]) # Y축 정렬
        return bubbles

    def _extract_text_ocr(self, img_rgb: np.ndarray, bbox: Tuple[int,int,int,int]) -> str:
        """OCR with Padding (인식률 향상)"""
        if not self.ocr_available: return ""
        import pytesseract
        
        x1, y1, x2, y2 = bbox
        h, w = img_rgb.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        roi = img_rgb[y1:y2, x1:x2]
        if roi.size == 0: return ""
        
        # Padding 추가 (핵심 개선)
        roi = cv2.copyMakeBorder(roi, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        try:
            text = pytesseract.image_to_string(binary, lang=self.ocr_lang, config='--psm 6')
            return text.strip().replace('\n', ' ')
        except:
            return ""

    def _scan_floating_text(self, img_rgb: np.ndarray, bubbles: List[Dict]) -> List[Dict]:
        """
        말풍선 외 영역(배경) 스캔 -> Floating Text 감지
        """
        if not self.ocr_available: return []
        import pytesseract
        
        # 말풍선 영역 마스킹 (흰색으로 덮음)
        masked_img = img_rgb.copy()
        for b in bubbles:
            x1, y1, x2, y2 = b['bbox']
            cv2.rectangle(masked_img, (x1, y1), (x2, y2), (255, 255, 255), -1)
        
        # 전체 OCR 수행
        try:
            # PSM 3 (Fully automatic page segmentation) or 11 (Sparse text)
            data = pytesseract.image_to_data(masked_img, lang=self.ocr_lang, output_type=pytesseract.Output.DICT, config='--psm 3')
            
            floating_candidates = []
            n_boxes = len(data['text'])
            for i in range(n_boxes):
                text = data['text'][i].strip()
                if not text: continue
                
                # 신뢰도 체크
                conf = int(data['conf'][i])
                if conf < 40: continue 
                
                x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                
                # 분류 (Strict Mode)
                cls_res = self.classifier.classify(text, is_in_bubble=False) if self.classifier else {'is_dialogue': True}
                
                if cls_res['is_dialogue']:
                    floating_candidates.append({
                        'text': text,
                        'bbox': (x, y, x+w, y+h),
                        'confidence': conf / 100.0,
                        'classification': cls_res
                    })
            return floating_candidates
            
        except Exception as e:
            print(f"Floating scan error: {e}")
            return []

    def process(self, image_path: str) -> Dict:
        """
        단일 이미지 처리 - 파이프라인 호환용
        """
        # 기본 반환 구조 (Pipeline이 기대하는 구조)
        result = {
            'action': 'process',
            'has_bubble': False,
            'has_dialogue': False,
            'bubble_count': 0,
            'bubble_confidence': 0.0,
            'detection_method': 'none',
            'text_analysis': {
                'dialogue_count': 0, 
                'sfx_count': 0, 
                'meta_count': 0, 
                'total': 0, 
                'texts': []
            },
            'path': image_path
        }
        
        # 1. 이미지 로드
        img = self._imread_unicode(image_path)
        if img is None:
            result['action'] = 'error'
            return result
        img_rgb = self._preprocess_image(img)
        
        # 2. YOLO 감지
        bubbles = self._detect_bubbles_yolo(img_rgb)
        
        detected_texts = [] # {'text':..., 'bbox':..., 'classification':...}
        
        # 3. 말풍선 내부 OCR 및 분류
        if bubbles:
            result['has_bubble'] = True
            result['bubble_count'] = len(bubbles)
            result['bubble_confidence'] = max([b['confidence'] for b in bubbles])
            result['detection_method'] = 'yolo'
            
            if self.use_ocr:
                for b in bubbles:
                    text = self._extract_text_ocr(img_rgb, b['bbox'])
                    if not text: continue
                    
                    # 분류 (In-Bubble Mode)
                    cls_res = self.classifier.classify(text, is_in_bubble=True) if self.classifier else {'is_dialogue': True, 'type':'dialogue'}
                    
                    detected_texts.append({
                        'text': text,
                        'bbox': b['bbox'],
                        'classification': cls_res
                    })
        
        # 4. Floating Text 감지 (말풍선이 없거나 적을 때 보완)
        # 말풍선이 아예 없거나, 있어도 확실한 대화가 감지 안됐을 때 체크
        has_clear_dialogue = any(t['classification']['is_dialogue'] for t in detected_texts)
        
        if not has_clear_dialogue or len(bubbles) == 0:
            floating_texts = self._scan_floating_text(img_rgb, bubbles)
            if floating_texts:
                detected_texts.extend(floating_texts)
                if not result['has_bubble']:
                    result['detection_method'] = 'ocr_floating'

        # 5. 통계 및 Action 결정
        dialogue_count = sum(1 for t in detected_texts if t['classification']['is_dialogue'])
        sfx_count = sum(1 for t in detected_texts if t['classification']['type'] == 'sfx')
        meta_count = sum(1 for t in detected_texts if t['classification']['type'] == 'meta')
        
        result['has_dialogue'] = dialogue_count > 0
        result['text_analysis'] = {
            'dialogue_count': dialogue_count,
            'sfx_count': sfx_count,
            'meta_count': meta_count,
            'total': len(detected_texts),
            'texts': detected_texts
        }
        
        # Action Logic (Pipeline 호환)
        if result['has_dialogue']:
            result['action'] = 'process' # 번역 대상
        elif sfx_count > 0:
            result['action'] = 'skip_sfx_only' # 효과음만 있음 -> 원본 사용
        elif result['has_bubble']:
             # 말풍선은 있는데 텍스트 인식이 안됨 -> 안전하게 처리 or 스킵?
             # YOLO 신뢰도가 높으면 처리하는게 좋음 (글자가 작아서 인식 못한 경우)
             result['action'] = 'process' 
        else:
            result['action'] = 'skip_no_bubble'
            
        return result

    # 배치 처리는 Pipeline App에서 for loop 돌리므로 여기서 구현 안 해도 무방하나
    # 호환성을 위해 껍데기만 유지하거나 삭제 가능. (App은 process()만 호출함)
    def process_batch(self, image_paths: List[str]) -> Dict:
        results = {}
        for path in image_paths:
            results[path] = self.process(path)
        return {'results': results}
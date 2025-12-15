"""
웹툰 자동 처리 파이프라인 v4.0 (Google Gemini API)
1. PDF/PSD/PSB → PNG 변환
2. PNG → 컷 분리 (패널 단위, 양방향 여백 감지)
3. 컷 사전 분류 (YOLO + OCR + 텍스트 분류) - NEW
4. 말풍선 제거 (선택적 Google Gemini API 호출)

v4.0 수정:
- 사전 분류 기능 추가 (analyze_cuts_for_bubble)
- 선택적 API 호출 기능 추가 (remove_speech_bubbles_selective)
- API 비용 절감 (효과음/말풍선 없음 → 원본 복사)

사용법:
  pip install google-genai PyMuPDF Pillow numpy psd-tools ultralytics pytesseract
  python webtoon_pipeline_google.py input.pdf --api-key YOUR_GOOGLE_API_KEY
"""
import os
import sys
import time
import json
import shutil
from pathlib import Path
from datetime import datetime
from io import BytesIO
from PIL import Image
import numpy as np

# ==========================================
# 🔑 Google API 키 하드코딩 (여기에 입력)
# ==========================================
HARDCODED_API_KEY = "" 
# ==========================================

# Google GenAI
try:
    from google import genai
    from google.genai import types
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False
    print("⚠️ google-genai 미설치: pip install google-genai")

# PDF 처리
try:
    import fitz  # PyMuPDF
    HAS_FITZ = True
except ImportError:
    HAS_FITZ = False
    print("⚠️ PyMuPDF 미설치: pip install PyMuPDF")

# PSD/PSB 처리
try:
    from psd_tools import PSDImage
    HAS_PSD = True
except ImportError:
    HAS_PSD = False
    print("⚠️ psd-tools 미설치: pip install psd-tools")

# 말풍선 프로세서
try:
    from webtoon_bubble_processor import WebtoonBubbleProcessor
    HAS_BUBBLE_PROCESSOR = True
except ImportError:
    HAS_BUBBLE_PROCESSOR = False
    print("⚠️ webtoon_bubble_processor 미설치")


# 기본 프롬프트 (대사 말풍선만 제거 - 환각 방지 강화)
DEFAULT_PROMPT = """<instruction_set>
  <task_definition>
    You are a highly specialized image editing AI. Your mission is to perform **surgical removal** of speech bubbles while guaranteeing the **absolute preservation** of all character art and background details.
    The previous execution critically failed by deleting characters. You must correct this. **Your primary goal is to protect the characters.**
  </task_definition>

  <critical_constraints description="RULES THAT CANNOT BE BROKEN. VIOLATION = FAILURE.">
    1.  **CHARACTER PROTECTION IS PARAMOUNT:** Under NO circumstances shall any part of a character's body, hair, clothing, or face be removed, erased, blurred, or altered. The inpainting process must **never** replace character art with background texture.
    2.  **SFX & Background Text are SACRED:** Do not touch any stylized sound effect text (e.g., '쪽', '부스스', '쿵') or any text that is part of the background art (signs, labels). These are immutable parts of the image.
    3.  **NO Global Changes:** Do not alter the overall color, lighting, or composition of the image.
  </critical_constraints>

  <remove_targets description="Identify and remove ONLY these specific elements">
    1.  **Speech Bubble Layer:** The white or colored shapes that contain dialogue text.
    2.  **Bubble Borders & Tails:** The black outlines and the pointed tails connecting bubbles to characters.
    3.  **Standard Dialogue Text:** The standard font text inside the bubbles.
  </remove_targets>

  <inpainting_rules description="How to fill the removed areas SAFELY">
    1.  **Reveal, Don't Replace:** When a speech bubble covers a character, your job is to **reveal** the character art that would logically be underneath it. You must reconstruct the hidden parts of the character (hair, clothes, skin) based on the visible surrounding art. **Do NOT fill character areas with wall/bed patterns.**
    2.  **Background-Only Inpainting:** Only when a bubble is entirely over a simple background (e.g., a wall or sky) should you use texture synthesis to fill it with the surrounding background pattern.
    3.  **Contextual Intelligence:** Before inpainting a pixel, determine: "Is this pixel part of a character or the background?" If it's a character, reconstruct the character. If it's background, reconstruct the background.

  <output_format>
    Return ONLY the final, processed image where speech bubbles are gone, but characters and SFX are perfectly intact.
  </output_format>
</instruction_set>"""


class WebtoonPipelineGoogle:
    """Google Gemini API 기반 웹툰 처리 파이프라인"""
    
    # 이미지 생성/편집 지원 모델
    MODELS = {
        'flash': 'gemini-2.5-flash-image',           # GA, 안정적, 빠름
        'pro': 'gemini-3-pro-image-preview',         # 최신 고품질
    }
    
    # 지원 파일 형식
    SUPPORTED_FORMATS = {
        'pdf': ['.pdf'],
        'psd': ['.psd', '.psb'],
        'image': ['.png', '.jpg', '.jpeg', '.webp']
    }
    
    def __init__(self, api_key=None, model='flash', require_api=True):
        """
        Args:
            api_key: Google API 키
            model: 'flash' 또는 'pro'
            require_api: True면 API 키 필수, False면 변환/분리만 사용 가능
        """
        # API 키 우선순위: 인자 > 하드코딩 > 환경변수
        self.api_key = (
            api_key or 
            HARDCODED_API_KEY or 
            os.getenv('GOOGLE_API_KEY') or 
            os.getenv('GEMINI_API_KEY')
        )
        
        self.client = None
        self.model_name = self.MODELS.get(model, self.MODELS['flash'])
        
        # API가 필요한 경우에만 검증
        if require_api:
            if not self.api_key:
                raise ValueError("GOOGLE_API_KEY 필요. https://aistudio.google.com 에서 발급")
            
            if not HAS_GENAI:
                raise ImportError("google-genai 필요: pip install google-genai")
            
            self.client = genai.Client(api_key=self.api_key)
            print(f"✓ Google Gemini API 초기화 완료")
            print(f"  모델: {self.model_name}")
        else:
            print(f"✓ 파이프라인 초기화 (변환/분리 모드)")
        
        self.stats = {
            'start_time': None,
            'end_time': None,
            'input_files': 0,
            'png_files': 0,
            'cuts_total': 0,
            'bubbles_removed': 0,
            'api_calls': 0,
            'api_skipped': 0,
            'errors': []
        }
        
        # 말풍선 프로세서 (사전 분류용)
        self.bubble_processor = None
    
    def _get_file_type(self, file_path):
        """파일 타입 판별"""
        ext = Path(file_path).suffix.lower()
        for file_type, extensions in self.SUPPORTED_FORMATS.items():
            if ext in extensions:
                return file_type
        return None
    
    # ========================================
    # 단계 1: PDF/PSD/PSB → PNG 변환
    # ========================================
    
    def pdf_to_png(self, pdf_path, output_dir, dpi=300):
        """PDF → PNG 변환"""
        if not HAS_FITZ:
            raise ImportError("PyMuPDF 필요: pip install PyMuPDF")
        
        pdf_path = Path(pdf_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"  PDF: {pdf_path.name}")
        
        png_files = []
        
        doc = fitz.open(str(pdf_path))
        total_pages = len(doc)
        
        for page_num in range(total_pages):
            page = doc[page_num]
            zoom = dpi / 72
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            
            output_filename = f"{pdf_path.stem}_page_{page_num + 1:03d}.png"
            output_path = output_dir / output_filename
            pix.save(str(output_path))
            
            png_files.append(str(output_path))
            print(f"    [{page_num + 1}/{total_pages}] {output_filename}")
        
        doc.close()
        return png_files
    
    def psd_to_png(self, psd_path, output_dir):
        """PSD/PSB → PNG 변환"""
        if not HAS_PSD:
            raise ImportError("psd-tools 필요: pip install psd-tools")
        
        psd_path = Path(psd_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"  PSD/PSB: {psd_path.name}")
        
        psd = PSDImage.open(str(psd_path))
        
        # 합성 이미지 (모든 레이어 병합)
        composite = psd.composite()
        
        if composite.mode != 'RGB':
            composite = composite.convert('RGB')
        
        output_filename = f"{psd_path.stem}.png"
        output_path = output_dir / output_filename
        composite.save(str(output_path))
        
        print(f"    → {output_filename} ({composite.width}x{composite.height})")
        
        return [str(output_path)]
    
    def image_to_png(self, image_path, output_dir):
        """이미지 파일 → PNG 복사/변환"""
        image_path = Path(image_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"  이미지: {image_path.name}")
        
        img = Image.open(str(image_path))
        
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        output_filename = f"{image_path.stem}.png"
        output_path = output_dir / output_filename
        img.save(str(output_path))
        
        print(f"    → {output_filename}")
        
        return [str(output_path)]
    
    def convert_to_png(self, input_files, output_dir, dpi=300):
        """다양한 형식 → PNG 변환"""
        print("\n" + "=" * 60)
        print("단계 1: 입력 파일 → PNG 변환")
        print("=" * 60)
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if isinstance(input_files, (str, Path)):
            input_files = [input_files]
        
        print(f"입력: {len(input_files)}개 파일")
        
        all_png_files = []
        
        for input_file in input_files:
            input_path = Path(input_file)
            
            if not input_path.exists():
                print(f"  ⚠️ 파일 없음: {input_path}")
                continue
            
            file_type = self._get_file_type(input_path)
            
            try:
                if file_type == 'pdf':
                    png_files = self.pdf_to_png(input_path, output_dir, dpi=dpi)
                elif file_type == 'psd':
                    png_files = self.psd_to_png(input_path, output_dir)
                elif file_type == 'image':
                    png_files = self.image_to_png(input_path, output_dir)
                else:
                    print(f"  ⚠️ 지원하지 않는 형식: {input_path.suffix}")
                    continue
                
                all_png_files.extend(png_files)
                self.stats['input_files'] += 1
                
            except Exception as e:
                print(f"  ⚠️ 변환 오류: {e}")
                self.stats['errors'].append(f"{input_path.name}: {e}")
        
        self.stats['png_files'] = len(all_png_files)
        
        print(f"\n✓ 완료: {len(all_png_files)}개 PNG")
        return all_png_files
    
    # ========================================
    # 단계 2: PNG → 컷 분리 (양방향 여백 감지)
    # ========================================
    
    def split_into_cuts(self, png_files, output_dir, 
                       min_gap_height=150, quality_threshold=0.5,
                       std_threshold=15, remove_empty_edges=False,
                       min_cut_height=200):
        """
        PNG → 컷 분리 (9:16 비율 기준 고정)
        
        9:16 = 가로:세로 비율
        최대 높이 = 너비 × (16/9)
        
        Args:
            min_gap_height: 최소 여백 높이 (기본: 150px) - 패널 간 구분선
            quality_threshold: 여백 품질 (기본: 0.5) - 미사용
            std_threshold: 균일 영역 표준편차 임계값 (기본: 15) - 미사용
            remove_empty_edges: 빈 여백 제거 (기본: False) - 미사용
            min_cut_height: 최소 컷 높이 (기본: 200px) - 미사용
        
        Note:
            9:16 비율 고정 - 최대 높이 = 너비 × 16/9
            이미지 훼손 방지 - 분리점에 여백 없으면 통째로 유지
        """
        print("\n" + "=" * 60)
        print("PNG → 컷 분리 (9:16 비율 기준)")
        print("=" * 60)
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 내부 여백 기준 (고정값)
        min_internal_gap = 10
        min_panel_gap = 40
        
        if isinstance(png_files, (str, Path)):
            png_files = [png_files]
        
        print(f"입력: {len(png_files)}개 파일")
        print(f"비율: 9:16 (최대 높이 = 너비 × 16/9)")
        print(f"내부 여백 기준: {min_internal_gap}px")
        print(f"패널 여백 기준: {min_panel_gap}px")
        
        all_cuts = []
        
        for file_idx, png_file in enumerate(png_files, 1):
            filename = Path(png_file).name
            print(f"\n[{file_idx}/{len(png_files)}] {filename}")
            
            try:
                img = Image.open(png_file)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                img_array = np.array(img)
                height, width = img_array.shape[:2]
                
                # 9:16 비율 기준 최대 높이 계산
                max_cut_height = int(width * 16 / 9)
                
                print(f"  크기: {width}x{height}")
                print(f"  9:16 기준 최대 높이: {max_cut_height}px")
                
                # 여백 감지
                all_gaps = self._detect_gaps_for_ratio(img_array, min_internal_gap)
                print(f"  감지된 여백: {len(all_gaps)}개")
                
                # 9:16 비율 기준 컷 분리
                cuts_boundaries = self._split_by_ratio(
                    all_gaps, height, max_cut_height, min_panel_gap
                )
                
                print(f"  분리 경계: {cuts_boundaries}")
                
                # 컷 저장
                page_stem = Path(png_file).stem
                cuts_saved = 0
                
                for i in range(len(cuts_boundaries) - 1):
                    start_y = cuts_boundaries[i]
                    end_y = cuts_boundaries[i + 1]
                    cut_height = end_y - start_y
                    
                    if cut_height < 30:  # 너무 작은 컷 스킵
                        continue
                    
                    cut_img = img.crop((0, start_y, width, end_y))
                    cuts_saved += 1
                    cut_filename = f"{page_stem}_cut_{cuts_saved:02d}.png"
                    cut_path = output_dir / cut_filename
                    cut_img.save(str(cut_path))
                    all_cuts.append(str(cut_path))
                    
                    status = "⚠️ 초과" if cut_height > max_cut_height else "✓"
                    print(f"    컷 {cuts_saved}: y={start_y}~{end_y} ({cut_height}px) {status}")
                
                print(f"  → {cuts_saved}개 컷 생성")
                
            except Exception as e:
                print(f"  ⚠️ 오류: {e}")
                self.stats['errors'].append(f"{filename}: {e}")
        
        self.stats['cuts_total'] = len(all_cuts)
        
        print(f"\n✓ 완료: 총 {len(all_cuts)}개 컷")
        return all_cuts
    
    def _detect_gaps_for_ratio(self, img_array, min_height=10):
        """
        9:16 비율 분리용 여백 감지
        
        Args:
            img_array: RGB 이미지 배열
            min_height: 최소 여백 높이
        
        Returns:
            여백 리스트 [{'start', 'end', 'mid', 'height', 'type'}, ...]
        """
        height = img_array.shape[0]
        
        # 그레이스케일 변환 및 행별 통계
        gray = np.mean(img_array, axis=2)
        row_brightness = np.mean(gray, axis=1)
        row_std = np.std(gray, axis=1)
        
        # 여백 판정 (어두운색: 밝기 < 50, 밝은색: 밝기 > 240, std < 15)
        is_dark_gap = (row_brightness < 50) & (row_std < 15)
        is_light_gap = (row_brightness > 240) & (row_std < 15)
        is_gap = is_dark_gap | is_light_gap
        
        # 연속 여백 영역 찾기
        gaps = []
        in_gap = False
        gap_start = 0
        
        for i in range(height):
            if is_gap[i] and not in_gap:
                gap_start = i
                in_gap = True
            elif not is_gap[i] and in_gap:
                gap_height = i - gap_start
                if gap_height >= min_height:
                    avg_brightness = np.mean(row_brightness[gap_start:i])
                    gaps.append({
                        'start': gap_start,
                        'end': i - 1,
                        'mid': (gap_start + i - 1) // 2,
                        'height': gap_height,
                        'type': 'black' if avg_brightness < 50 else 'white'
                    })
                in_gap = False
        
        # 마지막 여백 처리
        if in_gap:
            gap_height = height - gap_start
            if gap_height >= min_height:
                avg_brightness = np.mean(row_brightness[gap_start:height])
                gaps.append({
                    'start': gap_start,
                    'end': height - 1,
                    'mid': (gap_start + height - 1) // 2,
                    'height': gap_height,
                    'type': 'black' if avg_brightness < 50 else 'white'
                })
        
        return gaps
    
    def _split_by_ratio(self, all_gaps, total_height, max_cut_height, min_panel_gap):
        """
        비율 기준으로 컷 분리 경계 계산
        
        로직:
        1. 현재 위치에서 max_cut_height 범위 내의 여백들 확인
        2. 범위 내 마지막 여백(가장 아래쪽)에서 분리
        3. 여백이 없으면 고정 높이로 강제 분할
        
        Args:
            all_gaps: 모든 여백 리스트
            total_height: 전체 이미지 높이
            max_cut_height: 최대 컷 높이 (9:16 기준)
            min_panel_gap: 패널 구분 최소 여백
        
        Returns:
            분리 경계점 리스트
        """
        # 여백이 없으면 고정 높이로 분할
        if not all_gaps:
            print(f"    ⚠️ 여백 없음, 고정 높이({max_cut_height}px)로 분할")
            boundaries = [0]
            y = max_cut_height
            while y < total_height:
                boundaries.append(y)
                y += max_cut_height
            boundaries.append(total_height)
            return boundaries
        
        # 패널 구분용 여백만 필터링 (큰 여백)
        panel_gaps = [g for g in all_gaps if g['height'] >= min_panel_gap]
        
        # 상단 여백 처리: 첫 여백이 맨 위에 있으면 그 이후부터 시작
        start_y = 0
        if panel_gaps and panel_gaps[0]['start'] < 50:
            start_y = panel_gaps[0]['end'] + 1
            panel_gaps = panel_gaps[1:]
        
        # 하단 여백 처리: 마지막 여백이 맨 아래면 그 이전까지
        end_y = total_height
        if panel_gaps and panel_gaps[-1]['end'] > total_height - 50:
            end_y = panel_gaps[-1]['start']
            panel_gaps = panel_gaps[:-1]
        
        # 분리 경계 계산
        boundaries = [start_y]
        current_pos = start_y
        
        while current_pos < end_y:
            # 현재 위치에서 max_cut_height 범위 계산
            target_end = min(current_pos + max_cut_height, end_y)
            
            # 남은 높이가 max_cut_height 이하면 끝까지 포함
            if end_y - current_pos <= max_cut_height:
                boundaries.append(end_y)
                break
            
            # 범위 내 여백 찾기 (현재 위치 이후, target_end 이전)
            gaps_in_range = [g for g in all_gaps 
                           if g['mid'] > current_pos and g['mid'] <= target_end]
            
            if gaps_in_range:
                # 범위 내 가장 아래쪽 여백에서 분리
                best_gap = max(gaps_in_range, key=lambda g: g['mid'])
                split_point = best_gap['mid']
                boundaries.append(split_point)
                current_pos = split_point
            else:
                # 범위 내 여백 없음 - 다음 여백 찾기
                gaps_after = [g for g in all_gaps 
                             if g['mid'] > target_end]
                
                if gaps_after:
                    # 다음 여백까지 확장 (최대 2배까지만)
                    next_gap = min(gaps_after, key=lambda g: g['mid'])
                    if next_gap['mid'] - current_pos <= max_cut_height * 2:
                        split_point = next_gap['mid']
                        boundaries.append(split_point)
                        current_pos = split_point
                        print(f"    ⚠️ 범위 내 여백 없음, 다음 여백까지 확장: {split_point}px")
                    else:
                        # 너무 멀면 고정 높이로 분할
                        split_point = current_pos + max_cut_height
                        boundaries.append(split_point)
                        current_pos = split_point
                        print(f"    ⚠️ 여백 너무 멂, 고정 높이로 분할: {split_point}px")
                else:
                    # 여백이 전혀 없음 - 끝까지 포함
                    boundaries.append(end_y)
                    print(f"    ⚠️ 남은 여백 없음, 끝까지 포함")
                    break
        
        # 마지막 경계 추가
        if boundaries[-1] != end_y:
            boundaries.append(end_y)
        
        return boundaries
    
    # ========================================
    # 단계 3: 컷 사전 분류 (NEW)
    # ========================================
    
    def init_bubble_processor(self, model_path=None, confidence_threshold=0.15,
                               use_heuristic=True, use_ocr=True, use_text_filter=True):
        """말풍선 프로세서 초기화"""
        if not HAS_BUBBLE_PROCESSOR:
            print("⚠️ webtoon_bubble_processor를 사용할 수 없습니다.")
            return False
        
        try:
            self.bubble_processor = WebtoonBubbleProcessor(
                model_path=model_path,
                confidence_threshold=confidence_threshold,
                use_heuristic=use_heuristic,
                use_ocr=use_ocr,
                use_text_filter=use_text_filter
            )
            print("✓ 말풍선 프로세서 초기화 완료")
            return True
        except Exception as e:
            print(f"⚠️ 말풍선 프로세서 초기화 실패: {e}")
            return False
    
    def analyze_cuts_for_bubble(self, cuts_dir, model_path=None, 
                                 confidence_threshold=0.15,
                                 use_heuristic=True,
                                 use_ocr=True,
                                 use_text_filter=True,
                                 verbose=True):
        """
        컷 디렉토리 분석 (사전 분류)
        
        Args:
            cuts_dir: 컷 이미지가 있는 디렉토리
            model_path: YOLO 모델 경로
            confidence_threshold: 감지 임계값
            use_heuristic: 휴리스틱 보조 감지 사용
            use_ocr: OCR 사용
            use_text_filter: 텍스트 필터링 사용
            verbose: 상세 출력
        
        Returns:
            {
                'process': [파일명...],           # API 처리 대상
                'skip_sfx_only': [파일명...],     # 효과음만 (원본 복사)
                'skip_no_bubble': [파일명...],    # 말풍선 없음 (원본 복사)
                'skip_no_text': [파일명...],      # 텍스트 없음 (원본 복사)
                'details': {파일명: {...}, ...},  # 상세 분석 결과
                'stats': {...}                    # 통계
            }
        """
        print("\n" + "=" * 60)
        print("단계 3: 컷 사전 분류 (말풍선 감지)")
        print("=" * 60)
        
        cuts_dir = Path(cuts_dir)
        
        if not cuts_dir.exists():
            print(f"⚠️ 디렉토리 없음: {cuts_dir}")
            return None
        
        # 컷 파일 목록
        cut_files = sorted(cuts_dir.glob("*.png"))
        
        if not cut_files:
            print(f"⚠️ 컷 파일 없음: {cuts_dir}")
            return None
        
        print(f"분석 대상: {len(cut_files)}개 컷")
        
        # 프로세서 초기화
        if self.bubble_processor is None:
            if not self.init_bubble_processor(
                model_path=model_path,
                confidence_threshold=confidence_threshold,
                use_heuristic=use_heuristic,
                use_ocr=use_ocr,
                use_text_filter=use_text_filter
            ):
                # 프로세서 없으면 모두 처리 대상으로
                print("⚠️ 말풍선 프로세서 없음 - 모든 컷을 처리 대상으로 설정")
                return {
                    'process': [f.name for f in cut_files],
                    'skip_sfx_only': [],
                    'skip_no_bubble': [],
                    'skip_no_text': [],
                    'details': {f.name: {'action': 'process', 'reason': 'no_processor'} for f in cut_files},
                    'stats': {
                        'total': len(cut_files),
                        'to_process': len(cut_files),
                        'skip_sfx_only': 0,
                        'skip_no_bubble': 0,
                        'skip_no_text': 0,
                        'filter_rate': 0
                    }
                }
        
        # 분류 결과
        result = {
            'process': [],
            'skip_sfx_only': [],
            'skip_no_bubble': [],
            'skip_no_text': [],
            'details': {},
            'stats': {}
        }
        
        # 각 컷 분석
        for idx, cut_file in enumerate(cut_files):
            if verbose:
                print(f"  [{idx + 1}/{len(cut_files)}] {cut_file.name}...", end=" ")
            
            try:
                analysis = self.bubble_processor.process(str(cut_file))
                
                action = analysis.get('action', 'process')
                result['details'][cut_file.name] = {
                    'action': action,
                    'has_bubble': analysis.get('has_bubble', False),
                    'has_dialogue': analysis.get('has_dialogue', False),
                    'bubble_count': analysis.get('bubble_count', 0),
                    'bubble_confidence': analysis.get('bubble_confidence', 0),
                    'detection_method': analysis.get('detection_method', 'none'),
                    'text_analysis': analysis.get('text_analysis')
                }
                
                if action == 'process':
                    result['process'].append(cut_file.name)
                    if verbose:
                        ta = analysis.get('text_analysis', {})
                        print(f"✅ 처리대상 (대화:{ta.get('dialogue_count', 0)}개)")
                elif action == 'skip_sfx_only':
                    result['skip_sfx_only'].append(cut_file.name)
                    if verbose:
                        ta = analysis.get('text_analysis', {})
                        print(f"🔊 효과음만 ({ta.get('sfx_count', 0)}개)")
                elif action == 'skip_no_bubble':
                    result['skip_no_bubble'].append(cut_file.name)
                    if verbose:
                        print(f"⬜ 말풍선 없음")
                elif action == 'skip_no_text':
                    result['skip_no_text'].append(cut_file.name)
                    if verbose:
                        print(f"📝 텍스트 없음")
                else:
                    result['process'].append(cut_file.name)
                    if verbose:
                        print(f"❓ 기타 → 처리대상")
                        
            except Exception as e:
                # 오류 시 안전하게 처리 대상으로
                result['process'].append(cut_file.name)
                result['details'][cut_file.name] = {
                    'action': 'process',
                    'error': str(e)
                }
                if verbose:
                    print(f"⚠️ 오류 → 처리대상: {e}")
        
        # 통계
        total = len(cut_files)
        to_process = len(result['process'])
        skip_sfx = len(result['skip_sfx_only'])
        skip_no_bubble = len(result['skip_no_bubble'])
        skip_no_text = len(result['skip_no_text'])
        skip_total = skip_sfx + skip_no_bubble + skip_no_text
        filter_rate = (skip_total / total * 100) if total > 0 else 0
        
        result['stats'] = {
            'total': total,
            'to_process': to_process,
            'skip_sfx_only': skip_sfx,
            'skip_no_bubble': skip_no_bubble,
            'skip_no_text': skip_no_text,
            'filter_rate': filter_rate
        }
        
        print(f"\n📊 분류 결과:")
        print(f"   ✅ 처리 대상: {to_process}개")
        print(f"   🔊 효과음만: {skip_sfx}개")
        print(f"   ⬜ 말풍선 없음: {skip_no_bubble}개")
        print(f"   📝 텍스트 없음: {skip_no_text}개")
        print(f"   💰 API 절감률: {filter_rate:.1f}%")
        
        return result
    
    def save_analysis_result(self, cuts_dir, analysis_result):
        """분석 결과 JSON 저장"""
        cuts_dir = Path(cuts_dir)
        analysis_file = cuts_dir.parent / "2_cuts_analysis.json"
        
        save_data = {
            'timestamp': datetime.now().isoformat(),
            'cuts_dir': str(cuts_dir),
            'classification': {
                'process': analysis_result['process'],
                'skip_sfx_only': analysis_result['skip_sfx_only'],
                'skip_no_bubble': analysis_result['skip_no_bubble'],
                'skip_no_text': analysis_result.get('skip_no_text', [])
            },
            'details': analysis_result['details'],
            'stats': analysis_result['stats'],
            'user_modified': False
        }
        
        with open(analysis_file, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)
        
        print(f"✓ 분석 결과 저장: {analysis_file}")
        return str(analysis_file)
    
    def load_analysis_result(self, cuts_dir):
        """분석 결과 JSON 로드"""
        cuts_dir = Path(cuts_dir)
        analysis_file = cuts_dir.parent / "2_cuts_analysis.json"
        
        if not analysis_file.exists():
            return None
        
        with open(analysis_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return {
            'process': data['classification']['process'],
            'skip_sfx_only': data['classification']['skip_sfx_only'],
            'skip_no_bubble': data['classification']['skip_no_bubble'],
            'skip_no_text': data['classification'].get('skip_no_text', []),
            'details': data['details'],
            'stats': data['stats'],
            'user_modified': data.get('user_modified', False)
        }
    
    # ========================================
    # 단계 4: 말풍선 제거 (선택적)
    # ========================================
    
    def remove_speech_bubbles(self, cut_files, output_dir, prompt=None, delay=1):
        """말풍선 제거 (API 호출)"""
        print("\n" + "=" * 60)
        print("단계 4: 말풍선 제거 (API)")
        print("=" * 60)
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if isinstance(cut_files, (str, Path)):
            cut_files = [cut_files]
        
        prompt_text = prompt or DEFAULT_PROMPT
        
        print(f"모델: {self.model_name}")
        print(f"처리: {len(cut_files)}개 컷")
        print(f"딜레이: {delay}초")
        
        results = []
        success_count = 0
        
        for i, cut_file in enumerate(cut_files, 1):
            cut_path = Path(cut_file)
            filename = cut_path.name
            
            print(f"\n[{i}/{len(cut_files)}] {filename}")
            
            try:
                start_time = time.time()
                
                # 이미지 로드 (바이트로 읽기)
                with open(cut_path, 'rb') as f:
                    img_data = f.read()
                
                # MIME 타입 결정
                mime_type = "image/png"
                if filename.lower().endswith('.jpg') or filename.lower().endswith('.jpeg'):
                    mime_type = "image/jpeg"
                
                # API 호출
                response = None
                last_error = None
                max_retries = 3
                
                for retry in range(max_retries):
                    try:
                        response = self.client.models.generate_content(
                            model=self.model_name,
                            contents=[
                                types.Content(
                                    role="user",
                                    parts=[
                                        types.Part.from_bytes(data=img_data, mime_type=mime_type),
                                        types.Part(text=prompt_text)
                                    ]
                                )
                            ],
                            config=types.GenerateContentConfig(
                                response_modalities=['IMAGE', 'TEXT']
                            )
                        )
                        break
                    except Exception as api_error:
                        last_error = api_error
                        error_str = str(api_error).lower()
                        
                        if '500' in error_str or 'internal' in error_str:
                            wait_time = (retry + 1) * 5
                            print(f"  ⚠️ 서버 오류, {wait_time}초 후 재시도 ({retry + 1}/{max_retries})")
                            time.sleep(wait_time)
                        else:
                            raise api_error
                
                if response is None:
                    raise last_error if last_error else Exception("API 응답 없음")
                
                elapsed = time.time() - start_time
                
                # 디버깅: 응답 구조 출력
                print(f"  [DEBUG] response type: {type(response)}")
                print(f"  [DEBUG] response attrs: {[a for a in dir(response) if not a.startswith('_')][:15]}")
                
                # 응답 처리 - 다양한 방식 시도
                result_image = None
                
                # 방법 1: response.parts 직접 접근 (최신 API)
                if hasattr(response, 'parts') and response.parts:
                    print(f"  [DEBUG] response.parts 발견: {len(response.parts)}개")
                    for part in response.parts:
                        print(f"  [DEBUG] part type: {type(part)}, attrs: {[a for a in dir(part) if not a.startswith('_')][:10]}")
                        # inline_data (snake_case)
                        if hasattr(part, 'inline_data') and part.inline_data:
                            print(f"  [DEBUG] inline_data 발견")
                            image_data = part.inline_data.data
                            result_image = Image.open(BytesIO(image_data))
                            break
                        # inlineData (camelCase)
                        if hasattr(part, 'inlineData') and part.inlineData:
                            print(f"  [DEBUG] inlineData 발견")
                            image_data = part.inlineData.data
                            result_image = Image.open(BytesIO(image_data))
                            break
                
                # 방법 2: response.candidates 접근 (기존 방식)
                if result_image is None and hasattr(response, 'candidates') and response.candidates:
                    print(f"  [DEBUG] candidates 발견: {len(response.candidates)}개")
                    candidate = response.candidates[0]
                    print(f"  [DEBUG] candidate attrs: {[a for a in dir(candidate) if not a.startswith('_')][:10]}")
                    
                    # finish_reason 확인 (IMAGE_RECITATION 등)
                    finish_reason = getattr(candidate, 'finish_reason', None)
                    finish_reason_str = str(finish_reason) if finish_reason else ''
                    print(f"  [DEBUG] finish_reason: {finish_reason_str}")
                    
                    # 이미지 생성 실패 케이스 - 원본 복사
                    # IMAGE_RECITATION: 저작권 관련
                    # PROHIBITED_CONTENT: 콘텐츠 정책 위반
                    # SAFETY: 안전 필터
                    # BLOCKLIST: 차단 목록
                    skip_reasons = ['IMAGE_RECITATION', 'PROHIBITED_CONTENT', 'SAFETY', 'BLOCKLIST', 'OTHER']
                    should_copy_original = (
                        candidate.content is None or 
                        any(reason in finish_reason_str for reason in skip_reasons)
                    )
                    
                    if should_copy_original:
                        print(f"  ⚠️ API 제한 ({finish_reason_str or 'content None'}) - 원본 복사")
                        out_filename = filename.replace('.png', '_nobubble.png')
                        out_path = output_dir / out_filename
                        shutil.copy2(cut_file, str(out_path))
                        
                        results.append({
                            'input': cut_file,
                            'output': str(out_path),
                            'success': True,
                            'time': time.time() - start_time,
                            'note': 'original_copy'
                        })
                        success_count += 1
                        continue
                    
                    # content.parts에서 이미지 추출
                    if hasattr(candidate, 'content') and candidate.content:
                        print(f"  [DEBUG] content attrs: {[a for a in dir(candidate.content) if not a.startswith('_')][:10]}")
                        if hasattr(candidate.content, 'parts') and candidate.content.parts:
                            print(f"  [DEBUG] content.parts 발견: {len(candidate.content.parts)}개")
                            for part in candidate.content.parts:
                                print(f"  [DEBUG] content.part type: {type(part)}")
                                # inline_data (snake_case)
                                if hasattr(part, 'inline_data') and part.inline_data:
                                    print(f"  [DEBUG] content.part.inline_data 발견")
                                    image_data = part.inline_data.data
                                    result_image = Image.open(BytesIO(image_data))
                                    break
                                # inlineData (camelCase)  
                                if hasattr(part, 'inlineData') and part.inlineData:
                                    print(f"  [DEBUG] content.part.inlineData 발견")
                                    image_data = part.inlineData.data
                                    result_image = Image.open(BytesIO(image_data))
                                    break
                                # text 확인
                                if hasattr(part, 'text') and part.text:
                                    print(f"  [DEBUG] text 발견: {part.text[:100]}...")
                
                # 방법 3: response.text가 있으면 이미지 생성 실패
                if result_image is None and hasattr(response, 'text') and response.text:
                    print(f"  [DEBUG] response.text: {response.text[:200]}...")
                
                if result_image is None:
                    raise ValueError("응답에 이미지 없음")
                
                out_filename = filename.replace('.png', '_nobubble.png')
                out_path = output_dir / out_filename
                result_image.save(str(out_path))
                
                print(f"  ✓ 완료: {out_filename} ({elapsed:.1f}초)")
                
                results.append({
                    'input': cut_file,
                    'output': str(out_path),
                    'success': True,
                    'time': elapsed
                })
                
                success_count += 1
                self.stats['bubbles_removed'] += 1
                self.stats['api_calls'] += 1
                
                if i < len(cut_files) and delay > 0:
                    time.sleep(delay)
                
            except Exception as e:
                err_msg = str(e)
                print(f"  ✗ 오류: {err_msg[:150]}")
                self.stats['errors'].append(f"{filename}: {err_msg}")
                results.append({
                    'input': cut_file,
                    'output': None,
                    'success': False,
                    'error': err_msg
                })
        
        print(f"\n✓ 완료: {success_count}/{len(cut_files)} 성공")
        return results
    
    def remove_speech_bubbles_selective(self, cuts_dir, output_dir, 
                                         classification, 
                                         prompt=None, delay=1,
                                         progress_callback=None):
        """
        분류 기반 선택적 말풍선 제거
        
        Args:
            cuts_dir: 컷 이미지 디렉토리
            output_dir: 출력 디렉토리
            classification: 분류 결과 (process, skip_sfx_only, skip_no_bubble 리스트)
            prompt: API 프롬프트
            delay: API 호출 간격
            progress_callback: 진행 콜백 함수 (current, total, filename, action)
        
        Returns:
            results: 처리 결과 리스트
        """
        print("\n" + "=" * 60)
        print("단계 4: 선택적 말풍선 제거")
        print("=" * 60)
        
        cuts_dir = Path(cuts_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        api_targets = classification.get('process', [])
        skip_sfx = classification.get('skip_sfx_only', [])
        skip_no_bubble = classification.get('skip_no_bubble', [])
        skip_no_text = classification.get('skip_no_text', [])
        skip_all = skip_sfx + skip_no_bubble + skip_no_text
        
        total_files = len(api_targets) + len(skip_all)
        
        print(f"API 호출 대상: {len(api_targets)}개")
        print(f"원본 복사 대상: {len(skip_all)}개")
        
        results = []
        current = 0
        
        # 1. 스킵 대상 → 원본 복사
        for skip_file in skip_all:
            current += 1
            src = cuts_dir / skip_file
            dst = output_dir / skip_file.replace('.png', '_nobubble.png')
            
            if src.exists():
                shutil.copy2(src, dst)
                results.append({
                    'input': str(src),
                    'output': str(dst),
                    'success': True,
                    'action': 'copy',
                    'time': 0
                })
                self.stats['api_skipped'] += 1
                print(f"  [{current}/{total_files}] {skip_file} → 복사")
                
                if progress_callback:
                    progress_callback(current, total_files, skip_file, 'copy')
        
        # 2. API 호출 대상 → Gemini API
        if api_targets:
            api_files = [str(cuts_dir / f) for f in api_targets]
            
            for idx, cut_file in enumerate(api_files):
                current += 1
                filename = Path(cut_file).name
                
                if progress_callback:
                    progress_callback(current, total_files, filename, 'api')
                
                try:
                    api_result = self.remove_speech_bubbles(
                        [cut_file], str(output_dir),
                        prompt=prompt,
                        delay=delay if idx < len(api_files) - 1 else 0
                    )
                    results.extend(api_result)
                except Exception as e:
                    results.append({
                        'input': cut_file,
                        'output': None,
                        'success': False,
                        'action': 'api',
                        'error': str(e)
                    })
        
        # 통계
        success_count = sum(1 for r in results if r.get('success', False))
        
        print(f"\n✓ 선택적 처리 완료:")
        print(f"   API 호출: {len(api_targets)}개")
        print(f"   원본 복사: {len(skip_all)}개")
        print(f"   성공: {success_count}/{total_files}개")
        
        return results
    
    # ========================================
    # 전체 파이프라인
    # ========================================
    
    def run(self, input_path, output_base_dir, 
            dpi=300, min_gap_height=150,
            quality_threshold=0.8, std_threshold=15,
            min_cut_height=200,
            remove_empty_edges=True,
            api_prompt=None, api_delay=1,
            use_pre_classification=True,
            yolo_model_path=None):
        """
        전체 파이프라인 실행
        
        Args:
            use_pre_classification: 사전 분류 사용 여부
            yolo_model_path: YOLO 모델 경로
        """
        self.stats['start_time'] = datetime.now()
        
        print("\n" + "=" * 60)
        print("🎨 웹툰 자동 처리 (Google Gemini API)")
        print("=" * 60)
        print(f"입력: {input_path}")
        print(f"출력: {output_base_dir}")
        print(f"모델: {self.model_name}")
        print(f"사전 분류: {'사용' if use_pre_classification else '미사용'}")
        
        output_base_dir = Path(output_base_dir)
        output_base_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # 단계 1: 입력 → PNG
            png_dir = output_base_dir / "1_png"
            png_files = self.convert_to_png(input_path, png_dir, dpi=dpi)
            
            # 단계 2: PNG → 컷 분리
            cuts_dir = output_base_dir / "2_cuts"
            cut_files = self.split_into_cuts(
                png_files, cuts_dir,
                min_gap_height=min_gap_height,
                quality_threshold=quality_threshold,
                std_threshold=std_threshold,
                min_cut_height=min_cut_height,
                remove_empty_edges=remove_empty_edges
            )
            
            # 단계 3: 사전 분류 (선택적)
            final_dir = output_base_dir / "3_final"
            
            if use_pre_classification and HAS_BUBBLE_PROCESSOR:
                analysis = self.analyze_cuts_for_bubble(
                    cuts_dir,
                    model_path=yolo_model_path
                )
                
                if analysis:
                    self.save_analysis_result(cuts_dir, analysis)
                    
                    # 단계 4: 선택적 말풍선 제거
                    results = self.remove_speech_bubbles_selective(
                        cuts_dir, final_dir,
                        classification=analysis,
                        prompt=api_prompt,
                        delay=api_delay
                    )
                else:
                    # 분석 실패 시 전체 처리
                    results = self.remove_speech_bubbles(
                        cut_files, final_dir,
                        prompt=api_prompt,
                        delay=api_delay
                    )
            else:
                # 사전 분류 미사용 시 전체 처리
                results = self.remove_speech_bubbles(
                    cut_files, final_dir,
                    prompt=api_prompt,
                    delay=api_delay
                )
            
            self.stats['end_time'] = datetime.now()
            
            self._save_report(output_base_dir, results)
            
            print("\n" + "=" * 60)
            print("✓ 파이프라인 완료!")
            print("=" * 60)
            self._print_summary()
            
            return results
            
        except Exception as e:
            self.stats['end_time'] = datetime.now()
            print(f"\n✗ 오류: {e}")
            raise
    
    def _save_report(self, output_dir, results):
        report = {
            'timestamp': datetime.now().isoformat(),
            'model': self.model_name,
            'stats': {
                'input_files': self.stats['input_files'],
                'png_files': self.stats['png_files'],
                'cuts_total': self.stats['cuts_total'],
                'bubbles_removed': self.stats['bubbles_removed'],
                'api_calls': self.stats['api_calls'],
                'api_skipped': self.stats['api_skipped'],
                'errors': len(self.stats['errors'])
            },
            'error_details': self.stats['errors']
        }
        
        report_path = output_dir / 'report.json'
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
    
    def _print_summary(self):
        print(f"\n📊 요약:")
        print(f"  모델: {self.model_name}")
        print(f"  입력: {self.stats['input_files']}개 파일")
        print(f"  PNG: {self.stats['png_files']}개")
        print(f"  컷: {self.stats['cuts_total']}개")
        print(f"  API 호출: {self.stats['api_calls']}개")
        print(f"  API 스킵: {self.stats['api_skipped']}개")
        print(f"  처리: {self.stats['bubbles_removed']}개")
        print(f"  오류: {len(self.stats['errors'])}개")
        
        if self.stats['start_time'] and self.stats['end_time']:
            duration = self.stats['end_time'] - self.stats['start_time']
            print(f"  시간: {duration}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='웹툰 자동 처리 (Google Gemini API)')
    parser.add_argument('input_path', help='입력 파일 (PDF/PSD/PSB/PNG)')
    parser.add_argument('-o', '--output', default='./webtoon_output')
    parser.add_argument('--api-key', help='Google AI Studio API 키')
    parser.add_argument('--model', choices=['flash', 'pro'], default='flash')
    parser.add_argument('--dpi', type=int, default=300)
    parser.add_argument('--min-gap', type=int, default=150, help='최소 여백 높이 (기본: 150px)')
    parser.add_argument('--min-cut', type=int, default=200, help='최소 컷 높이 (기본: 200px)')
    parser.add_argument('--quality', type=float, default=0.8)
    parser.add_argument('--std-threshold', type=int, default=15)
    parser.add_argument('--delay', type=int, default=1)
    parser.add_argument('--no-pre-classify', action='store_true', help='사전 분류 비활성화')
    parser.add_argument('--yolo-model', type=str, default=None, help='YOLO 모델 경로')
    
    args = parser.parse_args()
    
    pipeline = WebtoonPipelineGoogle(
        google_api_key=args.api_key,
        model=args.model
    )
    
    pipeline.run(
        input_path=args.input_path,
        output_base_dir=args.output,
        dpi=args.dpi,
        min_gap_height=args.min_gap,
        min_cut_height=args.min_cut,
        quality_threshold=args.quality,
        std_threshold=args.std_threshold,
        api_delay=args.delay,
        use_pre_classification=not args.no_pre_classify,
        yolo_model_path=args.yolo_model
    )


if __name__ == "__main__":
    main()

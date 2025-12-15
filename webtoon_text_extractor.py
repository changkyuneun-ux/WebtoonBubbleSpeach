import easyocr
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import os
import re
import numpy as np

class WebtoonTextExtractor:
    def __init__(self, languages=['ko', 'en'], gpu=True):
        # 모델을 메모리에 로드합니다.
        print("⏳ EasyOCR 모델 로딩 중...")
        self.reader = easyocr.Reader(languages, gpu=gpu)
        print(f"✅ EasyOCR 초기화 완료 (languages={languages}, gpu={gpu})")

    def load_image(self, image_path):
        """
        이미지를 읽어옵니다.
        기존의 강제 이진화(Threshold) 로직을 제거하여 
        밝은 색 텍스트나 얇은 폰트가 사라지는 문제를 방지했습니다.
        """
        # 한글 경로 인식 호환성을 위해 numpy로 읽어서 디코딩
        img_array = np.fromfile(str(image_path), np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        if img is None:
            return None
            
        # EasyOCR은 RGB 포맷을 선호하므로 BGR -> RGB 변환
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        return img
    def is_valid_text(self, text):
        """
        강화된 텍스트 필터링 함수입니다.
        오탐(노이즈)을 줄이기 위해 조건을 추가했습니다.
        """
        if not text: return False
        clean_text = text.strip()
        
        # 조건 1: 텍스트 길이가 너무 짧으면 노이즈일 확률이 높음 (최소 2글자 이상)
        if len(clean_text) < 2:
            return False

        # 조건 2: 한국 웹툰 특성상 유의미한 대사에는 한글이 포함되어야 함
        # 영문/숫자만으로 이루어진 짧은 텍스트(j4 등)는 오탐으로 간주하고 제외
        has_korean = bool(re.search(r'[가-힣]', clean_text))
        
        # 한글이 한 글자라도 있으면 유효하다고 판단
        # (필요시 영어 대사도 허용하려면 이 조건을 완화할 수 있음)
        return has_korean

    def extract_text_from_image(self, image_path):
        """
        이미지에서 텍스트를 추출하고 위치 순서대로 정렬하여 반환합니다.
        """
        print(f"🔄 이미지 분석 중: {image_path.name}")
        
        # 1. 이미지 로드 (전처리 제거됨)
        img = self.load_image(image_path)
        if img is None:
            print(f"    ⚠️ 이미지를 읽을 수 없습니다: {image_path}")
            return []

        # 2. 텍스트 인식 수행
        # paragraph=True: 가까운 줄들을 문단으로 자동 병합
        # y_ths=0.5: 줄 간격 허용 범위 (기본값 0.5, 필요시 조정 가능)
        try:
            results = self.reader.readtext(img, detail=1, paragraph=True, y_ths=0.5)
        except Exception as e:
            print(f"    ⚠️ OCR 수행 중 오류 발생: {e}")
            return []

        # 3. 정렬 로직 (웹툰 읽는 순서: 위->아래 우선, 그 다음 좌->우)
        # box 구조: [[min_x, min_y], [max_x, min_y], ...]
        # y좌표(높이)를 기준으로 오름차순 정렬
        sorted_results = sorted(results, key=lambda r: (r[0][0][1], r[0][0][0]))

        return sorted_results

    def process_directory(self, base_dir, target_title=None, target_episode=None):
        """
        지정된 디렉토리 구조를 순회하며 텍스트를 추출합니다.
        구조: Title / Episode / Source / Page / 2_cuts
        """
        base_path = Path(base_dir)
        if not base_path.exists():
            print(f"⚠️ 기본 경로를 찾을 수 없습니다: {base_path}")
            return

        print(f"📂 디렉토리 스캔 시작: {base_path}")

        # 1. Title 레벨 순회
        for title_dir in sorted(base_path.iterdir()):
            if not title_dir.is_dir() or (target_title and title_dir.name != target_title):
                continue
            
            # 2. Episode 레벨 순회
            for episode_dir in sorted(title_dir.iterdir()):
                if not episode_dir.is_dir() or (target_episode and episode_dir.name != target_episode):
                    continue
                
                # 3. Source 레벨 순회
                for source_dir in sorted(episode_dir.iterdir()):
                    if not source_dir.is_dir(): continue
                    
                    # 4. Page 레벨 순회
                    for page_dir in sorted(source_dir.iterdir()):
                        if not page_dir.is_dir(): continue
                        
                        cuts_dir = page_dir / "2_cuts"
                        if not cuts_dir.exists():
                            continue
                        
                        print(f"  ➡️ 컷 처리 중: {cuts_dir}")
                        
                        # 이미지 파일 찾기 (png, jpg, jpeg)
                        image_files = sorted(list(cuts_dir.glob("*.png")) + list(cuts_dir.glob("*.jpg")))
                        
                        for cut_file in image_files:
                            # ID 생성 (파일명 등 활용)
                            cut_no = cut_file.stem
                            full_id = f"{title_dir.name}_{episode_dir.name}_{source_dir.name}_{page_dir.name}_{cut_no}"
                            
                            # 텍스트 추출 실행
                            results = self.extract_text_from_image(cut_file)
                            
                            # 결과 필터링 및 출력
                            valid_results = []
                            if results:
                                for box, text in results:
                                    if self.is_valid_text(text):
                                        valid_results.append(text.strip())

                            if valid_results:
                                print(f"    📄 [{full_id}] 텍스트 {len(valid_results)}개 발견:")
                                for i, text in enumerate(valid_results, 1):
                                    print(f"      {i}. {text}")
                            else:
                                print(f"    ❌ [{full_id}] 유효한 텍스트 없음.")
                            
                            print("-" * 40) # 구분선

if __name__ == "__main__":
    # ▼▼▼ 사용자 설정 영역 ▼▼▼
    
    # 1. 이미지 폴더 경로 (본인의 경로로 수정하세요)
    BASE_DIR = Path.home() / "voicetoon_image"  
    # 예: "C:/Users/name/voicetoon_image" 또는 "/Users/name/voicetoon_image"

    # 2. 특정 작품이나 에피소드만 돌리고 싶으면 이름을 적으세요 (전체는 None)
    TARGET_TITLE = None     # 예: "MyWebtoon"
    TARGET_EPISODE = None   # 예: "Ep001"
    
    # ▲▲▲ 사용자 설정 영역 끝 ▲▲▲

    # 실행
    extractor = WebtoonTextExtractor(gpu=True) 
    extractor.process_directory(BASE_DIR, TARGET_TITLE, TARGET_EPISODE)
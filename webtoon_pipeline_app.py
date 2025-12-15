"""
말풍선 제거 및 컷 자동분리 Agent(DOBEDUB) v4.1
실행: streamlit run webtoon_pipeline_app.py

메뉴:
A. 컷 분리 및 말풍선 제거
   - 1단계: 파일업로드 → 변환 → 컷분리
   - 2단계: 분리컷 확인 → 사전분류 → 말풍선 제거
B. 컷 보정(선택)
   - 3단계: 웹툰 컷 조정
"""
import streamlit as st

st.set_page_config(
    page_title="DOBEDUB - 웹툰 말풍선 제거",
    page_icon="🎨",
    layout="wide"
)

# CSS 스타일 적용 (색상은 테마 자동 적용)
st.markdown("""
<style>
/* 페이지 타이틀 */
.main-title {
    font-size: 28px;
    font-weight: 700;
    margin-bottom: 5px;
    padding-bottom: 10px;
    border-bottom: 2px solid currentColor;
    opacity: 0.9;
}
/* 큰 제목 (1단계, 2단계) */
.step-title {
    font-size: 22px;
    font-weight: 600;
    margin-top: 20px;
    margin-bottom: 15px;
    padding: 8px 0;
    border-bottom: 1px solid currentColor;
    opacity: 0.85;
}
/* 중간 제목 (1-1, 1-2) */
.section-title {
    font-size: 17px;
    font-weight: 600;
    margin-top: 15px;
    margin-bottom: 10px;
    opacity: 0.9;
}
/* 소제목 */
.sub-title {
    font-size: 15px;
    font-weight: 500;
    margin-top: 10px;
    margin-bottom: 8px;
    opacity: 0.85;
}
/* 사이드바 스타일 */
section[data-testid="stSidebar"] {
    width: 280px !important;
}
section[data-testid="stSidebar"] .stRadio > div {
    gap: 8px;
}
section[data-testid="stSidebar"] .stRadio label {
    padding: 10px 15px !important;
    border-radius: 6px;
    margin-bottom: 5px;
    font-size: 14px;
}
/* 사이드바 제목 */
section[data-testid="stSidebar"] h3 {
    font-size: 15px;
    margin-top: 15px;
    margin-bottom: 10px;
}
</style>
""", unsafe_allow_html=True)

import os
import sys
import shutil
import re
from pathlib import Path
import zipfile
import io
import json
from datetime import datetime

current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# 컷 보정 모듈 import
try:
    from webtoon_pipeline_cut import run_cut_editor_ui
    HAS_CUT_EDITOR = True
except ImportError:
    HAS_CUT_EDITOR = False

# ==========================================
# API 키 하드코딩 (개발용)
# ==========================================
HARDCODED_API_KEY = ""  # 여기에 API 키 입력: "AIzaSy..."
# ==========================================

OUTPUT_BASE_DIR = Path.home() / "voicetoon_image"
API_USAGE_FILE = OUTPUT_BASE_DIR / ".api_usage.json"

FILE_FORMATS = {
    'pdf': {'extensions': ['pdf'], 'name': 'PDF', 'has_dpi': True},
    'psd': {'extensions': ['psd', 'psb'], 'name': 'PSD/PSB', 'has_dpi': False},
    'image': {'extensions': ['png', 'jpg', 'jpeg', 'webp'], 'name': '이미지', 'has_dpi': False}
}

ALL_EXTENSIONS = []
for fmt in FILE_FORMATS.values():
    ALL_EXTENSIONS.extend(fmt['extensions'])

# 파이프라인 모듈 로드
try:
    from webtoon_pipeline_google import WebtoonPipelineGoogle, DEFAULT_PROMPT
    PIPELINE_OK = True
except ImportError as e:
    PIPELINE_OK = False
    PIPELINE_ERROR = str(e)
    DEFAULT_PROMPT = ""  # fallback

# 말풍선 프로세서 확인
try:
    from webtoon_bubble_processor import WebtoonBubbleProcessor
    BUBBLE_PROCESSOR_OK = True
except ImportError:
    BUBBLE_PROCESSOR_OK = False


# ==========================================
# 유틸리티 함수
# ==========================================
def get_title_list():
    """작품 목록 조회"""
    if not OUTPUT_BASE_DIR.exists():
        OUTPUT_BASE_DIR.mkdir(parents=True, exist_ok=True)
        return []
    return sorted([d.name for d in OUTPUT_BASE_DIR.iterdir() if d.is_dir() and not d.name.startswith('.')])


def get_episode_list(title):
    """회차 목록 조회"""
    if not title:
        return []
    title_dir = OUTPUT_BASE_DIR / title
    if not title_dir.exists():
        return []
    return sorted([d.name for d in title_dir.iterdir() if d.is_dir() and not d.name.startswith('.')])


def create_title(title_name):
    """새 작품 디렉토리 생성"""
    if title_name and title_name.strip():
        title_dir = OUTPUT_BASE_DIR / title_name.strip()
        title_dir.mkdir(parents=True, exist_ok=True)
        return True
    return False


def create_episode(title_name, episode_name):
    """새 회차 디렉토리 생성"""
    if title_name and episode_name and episode_name.strip():
        episode_dir = OUTPUT_BASE_DIR / title_name / episode_name.strip()
        episode_dir.mkdir(parents=True, exist_ok=True)
        return True
    return False


def load_api_usage():
    """API 사용량 로드"""
    if API_USAGE_FILE.exists():
        try:
            return json.loads(API_USAGE_FILE.read_text())
        except:
            pass
    return {'flash': 0, 'pro': 0}


def increment_api_usage(model, count=1):
    """API 사용량 증가"""
    usage = load_api_usage()
    usage[model] = usage.get(model, 0) + count
    API_USAGE_FILE.parent.mkdir(parents=True, exist_ok=True)
    API_USAGE_FILE.write_text(json.dumps(usage))
    return usage


# ==========================================
# Session State 초기화
# ==========================================
if 'file_uploader_key' not in st.session_state:
    st.session_state.file_uploader_key = 0
if 'selected_title' not in st.session_state:
    st.session_state.selected_title = None
if 'selected_episode' not in st.session_state:
    st.session_state.selected_episode = None
if 'prev_title' not in st.session_state:
    st.session_state.prev_title = None
if 'prev_episode' not in st.session_state:
    st.session_state.prev_episode = None

# 1단계 상태
if 'step1_done' not in st.session_state:
    st.session_state.step1_done = False
if 'converted_png_list' not in st.session_state:
    st.session_state.converted_png_list = []
if 'selected_png_indices' not in st.session_state:
    st.session_state.selected_png_indices = []
if 'processed_png_indices' not in st.session_state:
    st.session_state.processed_png_indices = []  # 처리 완료된 파일 인덱스
if 'cut_split_done' not in st.session_state:
    st.session_state.cut_split_done = False
if 'cut_info_list' not in st.session_state:
    st.session_state.cut_info_list = []
if 'conversion_info' not in st.session_state:
    st.session_state.conversion_info = {}

# 2단계 상태
if 'step2_classification_done' not in st.session_state:
    st.session_state.step2_classification_done = False
if 'cut_classification' not in st.session_state:
    st.session_state.cut_classification = {
        'process': [],
        'skip_sfx_only': [],
        'skip_no_bubble': [],
        'skip_no_text': [],
        'skip_bubble_only_cut': []  # 말풍선만 있는 컷
    }
if 'cut_details' not in st.session_state:
    st.session_state.cut_details = {}
if 'processing_done' not in st.session_state:
    st.session_state.processing_done = False
if 'processing_result' not in st.session_state:
    st.session_state.processing_result = None
if 'error_cuts' not in st.session_state:
    st.session_state.error_cuts = []
if 'preview_page_idx' not in st.session_state:
    st.session_state.preview_page_idx = None
if 'reprocess_selected' not in st.session_state:
    st.session_state.reprocess_selected = []


# ==========================================
# 단계별 초기화 함수
# ==========================================
def clear_from_png_conversion():
    """PNG 변환 이후 모든 결과 초기화"""
    st.session_state.step1_done = False
    st.session_state.converted_png_list = []
    st.session_state.selected_png_indices = []
    st.session_state.processed_png_indices = []
    clear_from_cut_split()

def clear_from_cut_split():
    """컷 분리 이후 모든 결과 초기화"""
    st.session_state.cut_split_done = False
    st.session_state.cut_info_list = []
    st.session_state.preview_page_idx = None
    clear_from_classification()

def clear_from_classification():
    """사전분류 이후 모든 결과 초기화"""
    st.session_state.step2_classification_done = False
    st.session_state.cut_classification = {
        'process': [],
        'skip_sfx_only': [],
        'skip_no_bubble': [],
        'skip_no_text': [],
        'skip_bubble_only_cut': []
    }
    st.session_state.cut_details = {}
    clear_from_processing()

def clear_from_processing():
    """말풍선 제거 결과 초기화"""
    st.session_state.processing_done = False
    st.session_state.processing_result = None
    st.session_state.error_cuts = []
    st.session_state.reprocess_selected = []
    # 재처리 체크박스 버전 증가 (새 체크박스 생성)
    st.session_state['reprocess_checkbox_version'] = st.session_state.get('reprocess_checkbox_version', 0) + 1


# ==========================================
# 메인 UI
# ==========================================
st.markdown(
    "<div class='main-title'>DOBEDUB - 웹툰 말풍선 제거 및 컷 자동분리</div>",
    unsafe_allow_html=True
)

if not PIPELINE_OK:
    st.error(f"모듈 로드 실패: {PIPELINE_ERROR}")
    st.stop()

# ==========================================
# 사이드바 - 메뉴 선택 및 API 설정
# ==========================================

# 이전 메뉴 상태 저장
if 'current_menu' not in st.session_state:
    st.session_state.current_menu = "A. 컷 분리 및 말풍선 제거"

with st.sidebar:
    st.markdown("### 작업 선택")
    app_mode = st.radio(
        "작업 선택",
        ["A. 컷 분리 및 말풍선 제거", "B. 컷 보정(선택)"],
        label_visibility="collapsed"
    )
    
    # 메뉴 변경 감지 및 상태 초기화
    if app_mode != st.session_state.current_menu:
        # 이전 메뉴 상태 초기화
        keys_to_reset = [
            'step1_done', 'converted_png_list', 'selected_png_indices', 
            'processed_png_indices', 'cut_split_done', 'cut_info_list',
            'step2_classification_done', 'cut_classification', 'cut_details',
            'processing_done', 'processing_result', 'error_cuts', 'conversion_info'
        ]
        for key in keys_to_reset:
            if key in st.session_state:
                del st.session_state[key]
        
        st.session_state.current_menu = app_mode
        st.rerun()
    
    st.divider()
    
    # API 키 설정
    st.markdown("### API 설정")
    
    # 기본 API 키 (하드코딩 또는 환경변수)
    default_api_key = HARDCODED_API_KEY or os.getenv('GOOGLE_API_KEY') or os.getenv('GEMINI_API_KEY') or ""
    
    if 'api_key' not in st.session_state:
        st.session_state.api_key = default_api_key
    
    if default_api_key:
        st.success("✓ API 키 설정됨")
        st.session_state.api_key = default_api_key
    else:
        input_key = st.text_input(
            "Google API Key",
            type="password",
            placeholder="AIzaSy...",
            key="api_key_input"
        )
        if input_key:
            st.session_state.api_key = input_key
            st.success("✓ API 키 입력됨")
    
    st.divider()

# ==========================================
# B. 컷 보정(선택) - 3단계
# ==========================================
if app_mode == "B. 컷 보정(선택)":
    st.markdown("<div class='step-title'>3. 웹툰 컷 조정</div>", unsafe_allow_html=True)
    st.info("※ 2단계까지가 정규 처리 단계입니다. 이 단계는 선택적으로 사용합니다.")
    st.divider()
    
    if HAS_CUT_EDITOR:
        run_cut_editor_ui()
    else:
        st.warning("webtoon_pipeline_cut.py 모듈을 찾을 수 없습니다.")
        st.info("webtoon_pipeline_cut.py 파일이 같은 디렉토리에 있어야 합니다.")
    st.stop()


# ==========================================
# A. 컷 분리 및 말풍선 제거
# ==========================================

# ==========================================
# 1단계: 파일업로드 → 변환 → 컷분리
# ==========================================
st.markdown("<div class='step-title'>1. 파일업로드 → 변환 → 컷분리</div>", unsafe_allow_html=True)

# 1-1. 파일 업로드
st.markdown("<div class='section-title'>1-1. 파일 업로드</div>", unsafe_allow_html=True)
st.caption("지원 형식: PDF(.pdf) | PSD/PSB(.psd, .psb) | 이미지(.png, .jpg, .jpeg, .webp)")

col_title, col_episode = st.columns(2)

with col_title:
    # 작품 리스트
    title_list = get_title_list()
    
    # 현재 선택된 작품의 인덱스 계산
    title_index = 0
    if st.session_state.selected_title and st.session_state.selected_title in title_list:
        title_index = title_list.index(st.session_state.selected_title) + 1
    
    title_options = ["선택하세요..."] + title_list
    selected_title = st.selectbox(
        "작품",
        options=title_options,
        index=title_index,
        key="title_select"
    )
    
    if selected_title == "선택하세요...":
        selected_title = None
    st.session_state.selected_title = selected_title
    
    # 작품 등록 (form 사용하여 rerun 방지)
    with st.form(key="new_title_form", clear_on_submit=True):
        new_title = st.text_input(
            "작품등록+",
            placeholder="새 작품명 입력 후 Enter",
            label_visibility="collapsed"
        )
        submitted = st.form_submit_button("등록", width='stretch')
        if submitted and new_title and new_title.strip():
            if new_title.strip() not in title_list:
                create_title(new_title.strip())
                st.session_state.selected_title = new_title.strip()
                st.rerun()

with col_episode:
    if selected_title:
        # 회차 리스트
        episode_list = get_episode_list(selected_title)
        
        # 현재 선택된 회차의 인덱스 계산
        episode_index = 0
        if st.session_state.selected_episode and st.session_state.selected_episode in episode_list:
            episode_index = episode_list.index(st.session_state.selected_episode) + 1
        
        episode_options = ["선택하세요..."] + episode_list
        selected_episode = st.selectbox(
            "회차",
            options=episode_options,
            index=episode_index,
            key="episode_select"
        )
        
        if selected_episode == "선택하세요...":
            selected_episode = None
        st.session_state.selected_episode = selected_episode
        
        # 회차 등록 (form 사용하여 rerun 방지)
        with st.form(key="new_episode_form", clear_on_submit=True):
            new_episode = st.text_input(
                "회차등록+",
                placeholder="새 회차명 입력 후 Enter",
                label_visibility="collapsed"
            )
            submitted = st.form_submit_button("등록", width='stretch')
            if submitted and new_episode and new_episode.strip():
                if new_episode.strip() not in episode_list:
                    create_episode(selected_title, new_episode.strip())
                    st.session_state.selected_episode = new_episode.strip()
                    st.rerun()
    else:
        st.selectbox("회차", ["작품을 먼저 선택하세요"], disabled=True, key="episode_disabled")
        st.text_input("회차등록+", disabled=True, placeholder="작품을 먼저 선택하세요", key="new_episode_disabled", label_visibility="collapsed")
        selected_episode = None
        st.session_state.selected_episode = None

# 작품/회차 변경 시 파일 목록 자동 초기화
if (st.session_state.prev_title != selected_title or 
    st.session_state.prev_episode != selected_episode):
    if st.session_state.prev_title is not None or st.session_state.prev_episode is not None:
        st.session_state.file_uploader_key += 1
        # 단계별 상태도 초기화
        st.session_state.step1_done = False
        st.session_state.cut_split_done = False
        st.session_state.step2_classification_done = False
        st.session_state.processing_done = False
    st.session_state.prev_title = selected_title
    st.session_state.prev_episode = selected_episode

# 파일 업로드
col_file_label, col_file_clear = st.columns([4, 1])
with col_file_label:
    if selected_title and selected_episode:
        st.info(f"저장 경로: `{OUTPUT_BASE_DIR / selected_title / selected_episode}/`")
with col_file_clear:
    if st.button("목록 초기화", key="clear_files"):
        st.session_state.file_uploader_key += 1
        st.rerun()

uploaded_files = st.file_uploader(
    "파일 선택",
    type=ALL_EXTENSIONS,
    accept_multiple_files=True,
    key=f"file_uploader_{st.session_state.file_uploader_key}",
    label_visibility="collapsed"
)

st.divider()

# 1-2. 파일 변환 설정
st.markdown("<div class='section-title'>1-2. 파일 변환 설정</div>", unsafe_allow_html=True)

# 업로드된 파일 유형 분석
pdf_files = []
psd_files = []
image_files = []

if uploaded_files:
    for f in uploaded_files:
        ext = Path(f.name).suffix.lower().lstrip('.')
        if ext in FILE_FORMATS['pdf']['extensions']:
            pdf_files.append(f)
        elif ext in FILE_FORMATS['psd']['extensions']:
            psd_files.append(f)
        elif ext in FILE_FORMATS['image']['extensions']:
            image_files.append(f)

# 파일 유형별 카운트 표시
if uploaded_files:
    file_summary = []
    if pdf_files:
        file_summary.append(f"PDF: {len(pdf_files)}개")
    if psd_files:
        file_summary.append(f"PSD/PSB: {len(psd_files)}개")
    if image_files:
        file_summary.append(f"이미지: {len(image_files)}개")
    
    if file_summary:
        st.info(" | ".join(file_summary))

# PDF DPI 설정 (PDF 파일이 있을 때만 표시)
pdf_dpi = 200  # 기본값
if pdf_files:
    pdf_dpi = st.selectbox("PDF 변환 해상도 (DPI)", [150, 200, 300], index=1, help="PDF → PNG 변환 시 해상도")
else:
    if uploaded_files:
        st.caption("ℹ️ PDF 파일이 없어 DPI 설정이 필요하지 않습니다.")

st.divider()

# 1-3. PNG 변환
can_convert = selected_title and selected_episode and uploaded_files
if not can_convert:
    st.info("작품, 회차, 파일을 모두 입력/선택하세요.")

if st.button("PNG 변환", type="primary", disabled=not can_convert, width='stretch'):
    # 기존 결과 초기화
    clear_from_png_conversion()
    
    with st.spinner("PNG 변환 중..."):
        episode_dir = OUTPUT_BASE_DIR / selected_title / selected_episode
        episode_dir.mkdir(parents=True, exist_ok=True)
        
        pipeline = WebtoonPipelineGoogle(require_api=False)  # 변환/분리는 API 불필요
        
        all_png_files = []
        
        progress = st.progress(0)
        status = st.empty()
        
        for idx, uploaded_file in enumerate(uploaded_files):
            status.info(f"[{idx+1}/{len(uploaded_files)}] {uploaded_file.name} 변환 중...")
            progress.progress((idx + 1) / len(uploaded_files))
            
            # 원본 파일 저장
            source_stem = Path(uploaded_file.name).stem
            source_dir = episode_dir / source_stem
            source_dir.mkdir(parents=True, exist_ok=True)
            
            temp_path = source_dir / uploaded_file.name
            temp_path.write_bytes(uploaded_file.getvalue())
            
            # PNG 변환
            temp_png_dir = source_dir / "_temp_png"
            temp_png_dir.mkdir(parents=True, exist_ok=True)
            
            try:
                png_files = pipeline.convert_to_png(
                    str(temp_path),
                    str(temp_png_dir),
                    dpi=pdf_dpi
                )
                
                for png_idx, png_file in enumerate(png_files):
                    all_png_files.append({
                        'index': len(all_png_files),
                        'source_file': uploaded_file.name,
                        'source_dir': str(source_dir),
                        'png_path': png_file,
                        'png_name': Path(png_file).name,
                        'png_stem': Path(png_file).stem
                    })
            except Exception as e:
                st.error(f"변환 오류 ({uploaded_file.name}): {e}")
        
        progress.empty()
        status.empty()
        
        if all_png_files:
            st.session_state.converted_png_list = all_png_files
            st.session_state.selected_png_indices = list(range(len(all_png_files)))
            st.session_state.step1_done = True
            st.session_state.conversion_info = {
                'title': selected_title,
                'episode': selected_episode,
                'episode_dir': str(episode_dir)
            }
            st.success(f"✅ {len(all_png_files)}개 PNG 변환 완료")
            st.rerun()

# 1-3. 컷분리 대상 선택
if st.session_state.step1_done and st.session_state.converted_png_list:
    st.divider()
    st.markdown("<div class='section-title'>1-3. 컷분리 대상 선택</div>", unsafe_allow_html=True)
    
    png_list = st.session_state.converted_png_list
    
    # 파일 상태 확인 함수
    def get_file_status(png_info):
        """파일 처리 상태 확인 (4_completion 또는 3_final 존재 여부)"""
        source_dir = Path(png_info['source_dir'])
        
        # page_XXX 패턴 추출
        page_match = re.search(r'page[_\s]*(\d+)', png_info['png_stem'], re.IGNORECASE)
        if page_match:
            page_dir_name = f"page_{page_match.group(1)}"
        else:
            num_match = re.search(r'(\d+)$', png_info['png_stem'])
            if num_match:
                page_dir_name = f"page_{num_match.group(1)}"
            else:
                page_dir_name = png_info['png_stem']
        
        page_dir = source_dir / page_dir_name
        completion_dir = page_dir / "4_completion"
        final_dir = page_dir / "3_final"
        cuts_dir = page_dir / "2_cuts"
        
        # 4_completion 확인
        if completion_dir.exists():
            files = list(completion_dir.glob("*.png"))
            if files:
                return {
                    'status': '완료',
                    'cut_count': len(files),
                    'result_dir': str(completion_dir),
                    'cuts_dir': str(cuts_dir),
                    'page_dir': str(page_dir),
                    'page_dir_name': page_dir_name
                }
        
        # 3_final 확인
        if final_dir.exists():
            files = list(final_dir.glob("*.png"))
            if files:
                return {
                    'status': '완료',
                    'cut_count': len(files),
                    'result_dir': str(final_dir),
                    'cuts_dir': str(cuts_dir),
                    'page_dir': str(page_dir),
                    'page_dir_name': page_dir_name
                }
        
        return {
            'status': '미처리',
            'cut_count': 0,
            'result_dir': None,
            'cuts_dir': str(cuts_dir),
            'page_dir': str(page_dir),
            'page_dir_name': page_dir_name
        }
    
    # 모든 파일의 상태 확인
    file_statuses = []
    for png_info in png_list:
        status_info = get_file_status(png_info)
        file_statuses.append({
            'png_info': png_info,
            **status_info
        })
    
    # 통계 계산
    completed_count = sum(1 for f in file_statuses if f['status'] == '완료')
    pending_count = len(file_statuses) - completed_count
    selected_count = len(st.session_state.selected_png_indices)
    
    # 통계 표시
    stat_cols = st.columns(4)
    with stat_cols[0]:
        st.metric("전체", f"{len(png_list)}개")
    with stat_cols[1]:
        st.metric("완료", f"{completed_count}개")
    with stat_cols[2]:
        st.metric("미처리", f"{pending_count}개")
    with stat_cols[3]:
        st.metric("선택됨", f"{selected_count}개")
    
    # 버튼 영역
    btn_cols = st.columns(3)
    with btn_cols[0]:
        if st.button("미처리만 선택", key="cut_sel_pending", width='stretch'):
            pending_indices = [
                f['png_info']['index'] for f in file_statuses if f['status'] == '미처리'
            ]
            st.session_state.selected_png_indices = pending_indices
            # 체크박스 session_state도 업데이트
            for f in file_statuses:
                idx = f['png_info']['index']
                st.session_state[f"tbl_sel_{idx}"] = idx in pending_indices
            st.session_state.cut_split_done = False
            st.session_state.step2_classification_done = False
            st.session_state.processing_done = False
            st.rerun()
    with btn_cols[1]:
        if st.button("전체 선택", key="cut_sel_all", width='stretch'):
            all_indices = [f['png_info']['index'] for f in file_statuses]
            st.session_state.selected_png_indices = all_indices
            # 체크박스 session_state도 업데이트
            for f in file_statuses:
                st.session_state[f"tbl_sel_{f['png_info']['index']}"] = True
            st.session_state.cut_split_done = False
            st.session_state.step2_classification_done = False
            st.session_state.processing_done = False
            st.rerun()
    with btn_cols[2]:
        if st.button("전체 해제", key="cut_sel_none", width='stretch'):
            st.session_state.selected_png_indices = []
            # 체크박스 session_state도 업데이트
            for f in file_statuses:
                st.session_state[f"tbl_sel_{f['png_info']['index']}"] = False
            st.session_state.cut_split_done = False
            st.session_state.step2_classification_done = False
            st.session_state.processing_done = False
            st.rerun()
    
    st.markdown("---")
    
    # 테이블 헤더
    header_cols = st.columns([0.5, 3, 1, 1, 1.5])
    with header_cols[0]:
        st.markdown("**작업**")
    with header_cols[1]:
        st.markdown("**파일명**")
    with header_cols[2]:
        st.markdown("**상태**")
    with header_cols[3]:
        st.markdown("**컷 수**")
    with header_cols[4]:
        st.markdown("**결과 조회**")
    
    st.markdown("<hr style='margin: 5px 0; border: none; border-top: 1px solid currentColor; opacity: 0.3;'>", unsafe_allow_html=True)
    
    # 파일 목록 테이블
    for file_info in file_statuses:
        png_info = file_info['png_info']
        global_idx = png_info['index']
        is_selected = global_idx in st.session_state.selected_png_indices
        is_completed = file_info['status'] == '완료'
        
        row_cols = st.columns([0.5, 3, 1, 1, 1.5])
        
        with row_cols[0]:
            new_selected = st.checkbox(
                "선택",
                value=is_selected,
                key=f"tbl_sel_{global_idx}",
                label_visibility="collapsed"
            )
        
        with row_cols[1]:
            # 페이지 번호만 표시
            st.text(file_info['page_dir_name'])
        
        with row_cols[2]:
            if is_completed:
                st.text("✓ 완료")
            else:
                st.text("미처리")
        
        with row_cols[3]:
            if is_completed:
                st.text(f"{file_info['cut_count']}개")
            else:
                st.text("-")
        
        with row_cols[4]:
            if is_completed:
                if st.button("조회", key=f"preview_{global_idx}", width='stretch'):
                    st.session_state.preview_page_idx = global_idx
                    st.rerun()
            else:
                st.button("조회", key=f"preview_disabled_{global_idx}", disabled=True, width='stretch')
        
        # 선택 상태 변경 처리
        if new_selected != is_selected:
            if new_selected:
                st.session_state.selected_png_indices.append(global_idx)
            else:
                if global_idx in st.session_state.selected_png_indices:
                    st.session_state.selected_png_indices.remove(global_idx)
            
            st.session_state.cut_split_done = False
            st.session_state.step2_classification_done = False
            st.session_state.processing_done = False
            st.session_state.cut_info_list = []
            st.session_state.cut_classification = {
                'process': [], 'skip_sfx_only': [], 'skip_no_bubble': [], 
                'skip_no_text': [], 'skip_bubble_only_cut': []
            }
            st.rerun()
    
    # 결과 조회 미리보기 섹션
    if st.session_state.preview_page_idx is not None:
        st.markdown("---")
        
        preview_idx = st.session_state.preview_page_idx
        preview_info = file_statuses[preview_idx]
        
        preview_header_cols = st.columns([4, 1])
        with preview_header_cols[0]:
            st.markdown(f"<div class='sub-title'>{preview_info['page_dir_name']} 결과 조회</div>", unsafe_allow_html=True)
        with preview_header_cols[1]:
            if st.button("닫기", key="close_preview", width='stretch'):
                st.session_state.preview_page_idx = None
                st.rerun()
        
        cuts_dir = Path(preview_info['cuts_dir'])
        result_dir = Path(preview_info['result_dir']) if preview_info['result_dir'] else None
        
        if result_dir and result_dir.exists():
            result_files = sorted(result_dir.glob("*.png"))
            cut_files = sorted(cuts_dir.glob("*.png")) if cuts_dir.exists() else []
            
            if result_files:
                # 컷 선택
                cut_names = [f.stem.replace("_nobubble", "") for f in result_files]
                selected_cut_idx = st.selectbox(
                    "컷 선택",
                    range(len(result_files)),
                    format_func=lambda x: f"{x+1}. {cut_names[x]}",
                    key="preview_cut_select"
                )
                
                # 원본/결과 비교
                compare_cols = st.columns(2)
                
                with compare_cols[0]:
                    st.markdown("**원본 (2_cuts)**")
                    # 원본 파일 찾기
                    cut_name = cut_names[selected_cut_idx]
                    original_file = cuts_dir / f"{cut_name}.png"
                    if original_file.exists():
                        # 캐시 방지를 위해 바이트로 읽기
                        with open(original_file, 'rb') as f:
                            st.image(f.read(), width='stretch')
                        st.caption(original_file.name)
                    else:
                        st.warning("원본 파일 없음")
                
                with compare_cols[1]:
                    st.markdown("**결과**")
                    result_file = result_files[selected_cut_idx]
                    # 캐시 방지를 위해 바이트로 읽기
                    with open(result_file, 'rb') as f:
                        st.image(f.read(), width='stretch')
                    st.caption(result_file.name)
        else:
            st.warning("결과 파일이 없습니다.")
    
    st.divider()
    
    # 컷 분리 실행
    selected_count = len(st.session_state.selected_png_indices)
    
    if st.button("✂️ 컷 분리 실행", type="primary", disabled=selected_count == 0, width='stretch'):
        # 기존 결과 초기화
        clear_from_cut_split()
        
        with st.spinner("컷 분리 중..."):
            pipeline = WebtoonPipelineGoogle(require_api=False)  # 컷 분리는 API 불필요
            
            cut_info_list = []
            
            progress = st.progress(0)
            status = st.empty()
            
            for i, idx in enumerate(st.session_state.selected_png_indices):
                png_info = png_list[idx]
                status.info(f"[{i+1}/{selected_count}] {png_info['png_name']} 컷 분리 중...")
                progress.progress((i + 1) / selected_count)
                
                # 출력 디렉토리 설정 (페이지 번호만 추출)
                source_dir = Path(png_info['source_dir'])
                
                # page_XXX 패턴 추출
                page_match = re.search(r'page[_\s]*(\d+)', png_info['png_stem'], re.IGNORECASE)
                if page_match:
                    page_dir_name = f"page_{page_match.group(1)}"
                else:
                    # 패턴 없으면 마지막 숫자 사용
                    num_match = re.search(r'(\d+)$', png_info['png_stem'])
                    if num_match:
                        page_dir_name = f"page_{num_match.group(1)}"
                    else:
                        page_dir_name = png_info['png_stem']
                
                page_dir = source_dir / page_dir_name
                cuts_dir = page_dir / "2_cuts"
                final_dir = page_dir / "3_final"
                
                cuts_dir.mkdir(parents=True, exist_ok=True)
                final_dir.mkdir(parents=True, exist_ok=True)
                
                try:
                    cut_files = pipeline.split_into_cuts(
                        [png_info['png_path']],
                        str(cuts_dir)
                    )
                    
                    cut_info_list.append({
                        'png_info': png_info,
                        'cuts_dir': str(cuts_dir),
                        'final_dir': str(final_dir),
                        'cut_files': cut_files,
                        'cut_count': len(cut_files)
                    })
                except Exception as e:
                    st.error(f"컷 분리 오류 ({png_info['png_name']}): {e}")
            
            progress.empty()
            status.empty()
            
            if cut_info_list:
                st.session_state.cut_info_list = cut_info_list
                st.session_state.cut_split_done = True
                
                total_cuts = sum(info['cut_count'] for info in cut_info_list)
                st.success(f"✅ 컷 분리 완료: {len(cut_info_list)}개 페이지 → {total_cuts}개 컷")
                st.rerun()


# ==========================================
# 2단계: 분리컷 확인 → 사전분류 → 말풍선 제거
# ==========================================
if st.session_state.cut_split_done and not st.session_state.processing_done:
    st.divider()
    st.markdown("<div class='step-title'>2. 분리컷 확인 → 사전분류 → 말풍선 제거</div>", unsafe_allow_html=True)
    
    cut_info_list = st.session_state.cut_info_list
    total_cuts = sum(info['cut_count'] for info in cut_info_list)
    
    # 2-1. 분리컷 확인
    st.markdown("<div class='section-title'>2-1. 분리컷 확인</div>", unsafe_allow_html=True)
    st.info(f"총 {total_cuts}개 컷 생성됨 ({len(cut_info_list)}개 페이지)")
    
    with st.expander("📋 분리된 컷 목록", expanded=False):
        for cut_info in cut_info_list:
            st.markdown(f"**{cut_info['png_info']['png_name']}** → {cut_info['cut_count']}개 컷")
            for cut_file in cut_info['cut_files']:
                st.text(f"  - {Path(cut_file).name}")
    
    st.divider()
    
    # 2-2. 사전분류 설정
    st.markdown("<div class='section-title'>2-2. 사전분류 설정</div>", unsafe_allow_html=True)
    
    use_pre_classification = st.checkbox(
        "사전분류 사용 (API 비용 절감)", 
        value=BUBBLE_PROCESSOR_OK, 
        disabled=not BUBBLE_PROCESSOR_OK
    )
    
    # YOLO 모델 자동 감지
    def find_yolo_model():
        """YOLO 모델 파일 자동 검색 - .pt 파일 찾기"""
        model_names = [
            "yolov8m_seg_speech_bubble.pt",
            "speech_bubble_seg.pt",
            "bubble_seg.pt",
        ]
        
        # __file__을 resolve()하여 절대 경로로 변환 (streamlit 실행 시 중요)
        script_path = Path(__file__).resolve()
        script_dir = script_path.parent  # src/
        project_dir = script_dir.parent  # WEBTOON/
        
        # 검색 디렉토리
        search_dirs = [
            script_dir,  # src/
            script_dir / "models",  # src/models/
            project_dir / "models",  # WEBTOON/models/
            Path.cwd() / "models",  # 현재디렉토리/models/
            Path.cwd(),  # 현재 작업 디렉토리
        ]
        
        # 특정 파일명 검색
        for model_name in model_names:
            for search_dir in search_dirs:
                path = search_dir / model_name
                if path.exists():
                    return str(path)
        
        # models 디렉토리에서 .pt 파일 검색
        for search_dir in search_dirs:
            if search_dir.exists() and search_dir.is_dir():
                pt_files = list(search_dir.glob("*.pt"))
                if pt_files:
                    for pt_file in pt_files:
                        name_lower = pt_file.name.lower()
                        if 'seg' in name_lower or 'bubble' in name_lower or 'speech' in name_lower:
                            return str(pt_file)
                    return str(pt_files[0])
        
        return None
    
    detected_yolo_path = find_yolo_model()
    
    if use_pre_classification:
        col_cls1, col_cls2, col_cls3 = st.columns(3)
        with col_cls1:
            confidence_threshold = st.slider("감지 임계값", 0.1, 0.5, 0.15, 0.05)
        with col_cls2:
            use_heuristic = st.checkbox("휴리스틱 보조 감지", value=True)
        with col_cls3:
            if detected_yolo_path:
                st.success(f"✓ YOLO 모델 감지됨")
                st.caption(f"`{Path(detected_yolo_path).name}`")
                yolo_model_path = detected_yolo_path
            else:
                yolo_model_path = st.text_input("YOLO 모델 경로", value="", placeholder="모델 파일 경로 입력")
                if not yolo_model_path:
                    st.warning("YOLO 모델 없음")
    
    st.divider()
    
    # 2-3. 모델 설정
    st.markdown("<div class='section-title'>2-3. 모델 설정</div>", unsafe_allow_html=True)
    
    # 사이드바에서 설정한 API 키 사용
    api_key = st.session_state.get('api_key', '')
    
    col_api1, col_api2, col_api3 = st.columns(3)
    with col_api1:
        if api_key:
            st.success("✓ API 키 설정됨 (사이드바)")
        else:
            st.warning("사이드바에서 API 키를 입력하세요")
    with col_api2:
        model = st.selectbox("모델", ["flash", "pro"], format_func=lambda x: "Gemini Flash" if x == "flash" else "Gemini Pro")
    with col_api3:
        api_delay = st.selectbox("API 간격", [0, 1, 2, 3], index=1, format_func=lambda x: f"{x}초")
    
    # 프롬프트 설정
    with st.expander("프롬프트 설정", expanded=False):
        custom_prompt = st.text_area("커스텀 프롬프트", value=DEFAULT_PROMPT, height=200)
    
    st.divider()
    
    # 사전분류 실행 또는 전체 처리
    if use_pre_classification and not st.session_state.step2_classification_done:
        if st.button("사전분류 실행", type="primary", width='stretch'):
            # 기존 결과 초기화
            clear_from_classification()
            
            with st.spinner("사전분류 진행 중..."):
                all_classification = {
                    'process': [],
                    'skip_sfx_only': [],
                    'skip_no_bubble': [],
                    'skip_no_text': [],
                    'skip_bubble_only_cut': []
                }
                all_details = {}
                
                progress = st.progress(0)
                status = st.empty()
                
                for i, cut_info in enumerate(cut_info_list):
                    status.info(f"[{i+1}/{len(cut_info_list)}] {cut_info['png_info']['png_name']} 분석 중...")
                    progress.progress((i + 1) / len(cut_info_list))
                    
                    try:
                        processor = WebtoonBubbleProcessor(
                            model_path=yolo_model_path if yolo_model_path else None,
                            confidence_threshold=confidence_threshold,
                            use_heuristic=use_heuristic,
                            use_ocr=True,
                            use_text_filter=True
                        )
                        
                        for cut_file in cut_info['cut_files']:
                            filename = Path(cut_file).name
                            key = f"{cut_info['png_info']['png_stem']}/{filename}"
                            
                            result = processor.process(cut_file)
                            
                            item = {
                                'key': key,
                                'filename': filename,
                                'cuts_dir': cut_info['cuts_dir'],
                                'final_dir': cut_info['final_dir'],
                                'png_stem': cut_info['png_info']['png_stem']
                            }
                            
                            action = result.get('action', 'process')
                            if action == 'process':
                                all_classification['process'].append(item)
                            elif action == 'skip_sfx_only':
                                all_classification['skip_sfx_only'].append(item)
                            elif action == 'skip_no_bubble':
                                all_classification['skip_no_bubble'].append(item)
                            elif action == 'skip_bubble_only_cut':
                                all_classification['skip_bubble_only_cut'].append(item)
                            else:
                                all_classification['skip_no_text'].append(item)
                            
                            all_details[key] = result
                            
                    except Exception as e:
                        # 오류 시 처리대상으로 분류
                        for cut_file in cut_info['cut_files']:
                            filename = Path(cut_file).name
                            key = f"{cut_info['png_info']['png_stem']}/{filename}"
                            all_classification['process'].append({
                                'key': key,
                                'filename': filename,
                                'cuts_dir': cut_info['cuts_dir'],
                                'final_dir': cut_info['final_dir'],
                                'png_stem': cut_info['png_info']['png_stem']
                            })
                
                progress.empty()
                status.empty()
                
                # 사전분류 결과를 각 페이지 디렉토리에 JSON으로 저장
                for cut_info in cut_info_list:
                    page_dir = Path(cut_info['cuts_dir']).parent
                    analysis_file = page_dir / "2_cuts_analysis.json"
                    
                    # 해당 페이지의 분류 결과 추출
                    page_stem = cut_info['png_info']['png_stem']
                    page_classification = {
                        'process': [],
                        'skip_sfx_only': [],
                        'skip_no_bubble': [],
                        'skip_no_text': [],
                        'skip_bubble_only_cut': []
                    }
                    page_details = {}
                    
                    for cat in ['process', 'skip_sfx_only', 'skip_no_bubble', 'skip_no_text', 'skip_bubble_only_cut']:
                        for item in all_classification[cat]:
                            if item.get('png_stem') == page_stem:
                                page_classification[cat].append(item['filename'])
                    
                    for key, detail in all_details.items():
                        if key.startswith(f"{page_stem}/"):
                            filename = key.split('/')[-1]
                            page_details[filename] = detail
                    
                    # 통계 계산
                    total = len(cut_info['cut_files'])
                    stats = {
                        'total': total,
                        'process': len(page_classification['process']),
                        'skip_sfx_only': len(page_classification['skip_sfx_only']),
                        'skip_no_bubble': len(page_classification['skip_no_bubble']),
                        'skip_no_text': len(page_classification['skip_no_text']),
                        'skip_bubble_only_cut': len(page_classification['skip_bubble_only_cut']),
                        'filter_rate': round((total - len(page_classification['process'])) / total * 100, 1) if total > 0 else 0
                    }
                    
                    analysis_data = {
                        'page': page_stem,
                        'classification': page_classification,
                        'details': page_details,
                        'stats': stats
                    }
                    
                    try:
                        with open(analysis_file, 'w', encoding='utf-8') as f:
                            json.dump(analysis_data, f, ensure_ascii=False, indent=2)
                    except Exception as e:
                        st.warning(f"분석 결과 저장 실패 ({page_stem}): {e}")
                
                st.session_state.cut_classification = all_classification
                st.session_state.cut_details = all_details
                st.session_state.step2_classification_done = True
                st.rerun()
    
    elif not use_pre_classification and not st.session_state.step2_classification_done:
        st.warning("사전분류 비활성화 - 모든 컷을 API 처리 대상으로 설정합니다.")
        
        if st.button("▶️ 전체 처리 대상으로 설정", type="primary", width='stretch'):
            all_classification = {
                'process': [],
                'skip_sfx_only': [],
                'skip_no_bubble': [],
                'skip_no_text': []
            }
            
            for cut_info in cut_info_list:
                for cut_file in cut_info['cut_files']:
                    filename = Path(cut_file).name
                    key = f"{cut_info['png_info']['png_stem']}/{filename}"
                    all_classification['process'].append({
                        'key': key,
                        'filename': filename,
                        'cuts_dir': cut_info['cuts_dir'],
                        'final_dir': cut_info['final_dir'],
                        'png_stem': cut_info['png_info']['png_stem']
                    })
            
            st.session_state.cut_classification = all_classification
            st.session_state.step2_classification_done = True
            st.rerun()


# 2-4. 분류 결과 조정
if st.session_state.step2_classification_done and not st.session_state.processing_done:
    st.divider()
    st.markdown("<div class='section-title'>2-4. 분류 결과 조정</div>", unsafe_allow_html=True)
    
    classification = st.session_state.cut_classification
    
    process_count = len(classification['process'])
    sfx_count = len(classification['skip_sfx_only'])
    no_bubble_count = len(classification['skip_no_bubble'])
    no_text_count = len(classification['skip_no_text'])
    bubble_only_count = len(classification.get('skip_bubble_only_cut', []))
    total = process_count + sfx_count + no_bubble_count + no_text_count + bubble_only_count
    skip_total = sfx_count + no_bubble_count + no_text_count + bubble_only_count
    filter_rate = (skip_total / total * 100) if total > 0 else 0
    
    # 통계 표시
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1:
        st.metric("총 컷", f"{total}개")
    with col2:
        st.metric("처리대상", f"{process_count}개")
    with col3:
        st.metric("효과음", f"{sfx_count}개")
    with col4:
        st.metric("없음", f"{no_bubble_count + no_text_count}개")
    with col5:
        st.metric("풍선만", f"{bubble_only_count}개")
    with col6:
        st.metric("절감률", f"{filter_rate:.0f}%")
    
    # 말풍선만 있는 컷 경고
    if bubble_only_count > 0:
        st.warning(f"**말풍선만 있는 컷 {bubble_only_count}개 감지**: 이 컷은 캐릭터 없이 말풍선만 있습니다. 이전/다음 컷과 통합을 권장합니다.")
    
    # 분류별 표시 및 조정
    st.markdown("<div class='sub-title'>처리 대상 (API 호출)</div>", unsafe_allow_html=True)
    if classification['process']:
        cols = st.columns(6)
        for idx, item in enumerate(classification['process'][:12]):
            with cols[idx % 6]:
                img_path = Path(item['cuts_dir']) / item['filename']
                if img_path.exists():
                    st.image(str(img_path), width='stretch')
                st.caption(item['filename'])
                if st.button("→ 제외", key=f"to_skip_{item['key']}", width='stretch'):
                    classification['process'].remove(item)
                    classification['skip_sfx_only'].append(item)
                    st.session_state.cut_classification = classification
                    st.rerun()
        if len(classification['process']) > 12:
            st.caption(f"... 외 {len(classification['process']) - 12}개")
    else:
        st.caption("없음")
    
    # 말풍선만 있는 컷
    if classification.get('skip_bubble_only_cut'):
        st.markdown("<div class='sub-title'>말풍선만 있는 컷 (원본 복사 권장)</div>", unsafe_allow_html=True)
        cols = st.columns(6)
        for idx, item in enumerate(classification['skip_bubble_only_cut'][:6]):
            with cols[idx % 6]:
                img_path = Path(item['cuts_dir']) / item['filename']
                if img_path.exists():
                    st.image(str(img_path), width='stretch')
                st.caption(item['filename'])
                col_btn1, col_btn2 = st.columns(2)
                with col_btn1:
                    if st.button("→ 처리", key=f"boc_proc_{item['key']}", width='stretch'):
                        classification['skip_bubble_only_cut'].remove(item)
                        classification['process'].append(item)
                        st.session_state.cut_classification = classification
                        st.rerun()
                with col_btn2:
                    if st.button("→ 스킵", key=f"boc_skip_{item['key']}", width='stretch'):
                        classification['skip_bubble_only_cut'].remove(item)
                        classification['skip_no_bubble'].append(item)
                        st.session_state.cut_classification = classification
                        st.rerun()
    
    st.markdown("<div class='sub-title'>효과음/의성어 + 말풍선 없음 (원본 복사)</div>", unsafe_allow_html=True)
    skip_all = classification['skip_sfx_only'] + classification['skip_no_bubble'] + classification['skip_no_text']
    if skip_all:
        cols = st.columns(6)
        for idx, item in enumerate(skip_all[:12]):
            with cols[idx % 6]:
                img_path = Path(item['cuts_dir']) / item['filename']
                if img_path.exists():
                    st.image(str(img_path), width='stretch')
                st.caption(item['filename'])
                if st.button("→ 처리", key=f"to_proc_{item['key']}", width='stretch'):
                    # 해당 리스트에서 제거
                    for cat in ['skip_sfx_only', 'skip_no_bubble', 'skip_no_text']:
                        if item in classification[cat]:
                            classification[cat].remove(item)
                            break
                    classification['process'].append(item)
                    st.session_state.cut_classification = classification
                    st.rerun()
        if len(skip_all) > 12:
            st.caption(f"... 외 {len(skip_all) - 12}개")
    else:
        st.caption("없음")
    
    # 일괄 조정 버튼
    col_batch1, col_batch2, col_batch3 = st.columns(3)
    with col_batch1:
        if st.button("전체 → 처리대상", width='stretch'):
            for cat in ['skip_sfx_only', 'skip_no_bubble', 'skip_no_text']:
                classification['process'].extend(classification[cat])
                classification[cat] = []
            # 말풍선만 있는 컷도 포함
            if 'skip_bubble_only_cut' in classification:
                classification['process'].extend(classification['skip_bubble_only_cut'])
                classification['skip_bubble_only_cut'] = []
            st.session_state.cut_classification = classification
            st.rerun()
    with col_batch2:
        if st.button("📤 전체 → 제외", width='stretch'):
            classification['skip_sfx_only'].extend(classification['process'])
            classification['process'] = []
            # 말풍선만 있는 컷도 스킵으로 이동
            if 'skip_bubble_only_cut' in classification:
                classification['skip_no_bubble'].extend(classification['skip_bubble_only_cut'])
                classification['skip_bubble_only_cut'] = []
            st.session_state.cut_classification = classification
            st.rerun()
    with col_batch3:
        if st.button("분류 다시하기", width='stretch'):
            st.session_state.step2_classification_done = False
            st.session_state.cut_classification = {'process': [], 'skip_sfx_only': [], 'skip_no_bubble': [], 'skip_no_text': [], 'skip_bubble_only_cut': []}
            st.rerun()
    
    st.divider()
    
    # 말풍선 제거 실행
    if not api_key:
        st.warning("API 키를 입력하세요.")
    else:
        if st.button("말풍선 제거 실행", type="primary", width='stretch', disabled=process_count == 0 and skip_total == 0):
            # 기존 결과 초기화
            clear_from_processing()
            
            # 현재 설정 저장 (재처리 시 기본값으로 사용)
            st.session_state['last_custom_prompt'] = custom_prompt
            st.session_state['last_model'] = model
            
            with st.spinner("말풍선 제거 처리 중..."):
                start_time = datetime.now()
                
                pipeline = WebtoonPipelineGoogle(api_key=api_key, model=model)
                
                results = []
                error_cuts = []
                
                # 출력 디렉토리별 그룹화
                output_groups = {}
                for item in classification['process']:
                    final_dir = item['final_dir']
                    if final_dir not in output_groups:
                        output_groups[final_dir] = {'cuts_dir': item['cuts_dir'], 'process': [], 'skip': []}
                    output_groups[final_dir]['process'].append(item['filename'])
                
                for cat in ['skip_sfx_only', 'skip_no_bubble', 'skip_no_text', 'skip_bubble_only_cut']:
                    for item in classification.get(cat, []):
                        final_dir = item['final_dir']
                        if final_dir not in output_groups:
                            output_groups[final_dir] = {'cuts_dir': item['cuts_dir'], 'process': [], 'skip': []}
                        output_groups[final_dir]['skip'].append(item['filename'])
                
                total_items = process_count + skip_total
                current_item = 0
                success_count = 0
                
                progress = st.progress(0)
                status = st.empty()
                
                for final_dir, group in output_groups.items():
                    cuts_dir = Path(group['cuts_dir'])
                    final_path = Path(final_dir)
                    final_path.mkdir(parents=True, exist_ok=True)
                    
                    # 스킵 대상 복사
                    for filename in group['skip']:
                        current_item += 1
                        src = cuts_dir / filename
                        dst = final_path / filename.replace('.png', '_nobubble.png')
                        
                        if src.exists():
                            shutil.copy2(src, dst)
                            success_count += 1
                        
                        progress.progress(current_item / total_items)
                        status.info(f"[{current_item}/{total_items}] {filename}: 원본 복사")
                    
                    # API 호출 대상 처리
                    for filename in group['process']:
                        current_item += 1
                        cut_file = str(cuts_dir / filename)
                        
                        progress.progress(current_item / total_items)
                        status.info(f"[{current_item}/{total_items}] {filename}: API 처리 중...")
                        
                        try:
                            api_results = pipeline.remove_speech_bubbles(
                                [cut_file], str(final_path),
                                prompt=custom_prompt,
                                delay=api_delay
                            )
                            
                            if api_results:
                                for r in api_results:
                                    if r.get('success', False):
                                        success_count += 1
                                        results.append(r)
                                    else:
                                        error_cuts.append({
                                            'filename': filename,
                                            'cuts_dir': str(cuts_dir),
                                            'final_dir': final_dir,
                                            'error': r.get('error', 'Unknown error')
                                        })
                            else:
                                error_cuts.append({
                                    'filename': filename,
                                    'cuts_dir': str(cuts_dir),
                                    'final_dir': final_dir,
                                    'error': 'API 응답 없음'
                                })
                                    
                        except Exception as e:
                            error_cuts.append({
                                'filename': filename,
                                'cuts_dir': str(cuts_dir),
                                'final_dir': final_dir,
                                'error': str(e)
                            })
                
                progress.progress(100)
                status.empty()
                
                # 4_completion 자동 생성
                for final_dir in output_groups.keys():
                    final_path = Path(final_dir)
                    completion_dir = final_path.parent / "4_completion"
                    
                    if final_path.exists():
                        completion_dir.mkdir(parents=True, exist_ok=True)
                        for final_file in final_path.glob("*.png"):
                            new_name = final_file.name.replace("_nobubble", "")
                            shutil.copy2(final_file, completion_dir / new_name)
                
                end_time = datetime.now()
                elapsed_seconds = (end_time - start_time).total_seconds()
                
                # API 사용량 업데이트
                increment_api_usage(model, process_count)
                
                # 결과 저장
                st.session_state.processing_done = True
                st.session_state.processing_result = {
                    'total_cuts': total_items,
                    'api_calls': process_count,
                    'api_skipped': skip_total,
                    'success_count': success_count,
                    'error_count': len(error_cuts),
                    'elapsed_seconds': elapsed_seconds,
                    'model': model,
                    'filter_rate': filter_rate
                }
                st.session_state.error_cuts = error_cuts
                st.rerun()


# ==========================================
# 2-5. 처리 결과 리포트
# ==========================================
if st.session_state.processing_done:
    st.divider()
    st.markdown("<div class='section-title'>2-5. 처리 결과 리포트</div>", unsafe_allow_html=True)
    
    result = st.session_state.processing_result
    
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("총 컷", f"{result['total_cuts']}개")
    with col2:
        st.metric("API 호출", f"{result['api_calls']}회")
    with col3:
        st.metric("성공", f"{result['success_count']}개")
    with col4:
        st.metric("오류", f"{result['error_count']}개")
    with col5:
        elapsed = result['elapsed_seconds']
        if elapsed >= 60:
            st.metric("소요시간", f"{int(elapsed // 60)}분 {int(elapsed % 60)}초")
        else:
            st.metric("소요시간", f"{elapsed:.1f}초")
    
    st.info(f"API 절감률: **{result['filter_rate']:.1f}%** | 모델: {result['model']}")
    
    # 오류 컷 리스트 및 재처리
    error_cuts = st.session_state.error_cuts
    if error_cuts:
        st.markdown("<div class='sub-title'>오류 컷 리스트</div>", unsafe_allow_html=True)
        
        # 전체 선택/해제
        col_sel_all, col_sel_none = st.columns(2)
        with col_sel_all:
            if st.button("전체 선택", key="err_sel_all", width='stretch'):
                for idx in range(len(error_cuts)):
                    st.session_state[f"err_sel_{idx}"] = True
                st.rerun()
        with col_sel_none:
            if st.button("전체 해제", key="err_sel_none", width='stretch'):
                for idx in range(len(error_cuts)):
                    st.session_state[f"err_sel_{idx}"] = False
                st.rerun()
        
        for idx, err in enumerate(error_cuts):
            col_check, col_name, col_err = st.columns([0.5, 2, 4])
            with col_check:
                st.checkbox("선택", key=f"err_sel_{idx}", label_visibility="collapsed")
            with col_name:
                st.text(err['filename'])
            with col_err:
                st.caption(err['error'][:50] + "..." if len(err['error']) > 50 else err['error'])
        
        # 선택된 오류 컷 수집
        selected_errors = [err for idx, err in enumerate(error_cuts) 
                          if st.session_state.get(f"err_sel_{idx}", False)]
        
        if st.button(f"선택 컷 재처리 ({len(selected_errors)}개)", 
                    width='stretch', 
                    disabled=len(selected_errors) == 0):
            
            api_key = st.session_state.get('api_key', '')
            model = result.get('model', 'flash')
            
            if not api_key:
                st.error("API 키가 필요합니다.")
            else:
                with st.spinner(f"{len(selected_errors)}개 컷 재처리 중..."):
                    pipeline = WebtoonPipelineGoogle(api_key=api_key, model=model)
                    
                    retry_success = 0
                    retry_failed = []
                    
                    progress = st.progress(0)
                    
                    for i, err in enumerate(selected_errors):
                        progress.progress((i + 1) / len(selected_errors))
                        
                        try:
                            input_path = Path(err['cuts_dir']) / err['filename']
                            output_dir = err['final_dir']
                            
                            if input_path.exists():
                                api_results = pipeline.remove_speech_bubbles(
                                    [str(input_path)],
                                    output_dir,
                                    delay=1.0
                                )
                                
                                if api_results and api_results[0].get('success', False):
                                    retry_success += 1
                                else:
                                    retry_failed.append({
                                        'filename': err['filename'],
                                        'cuts_dir': err['cuts_dir'],
                                        'final_dir': err['final_dir'],
                                        'error': api_results[0].get('error', 'Unknown error') if api_results else 'No result'
                                    })
                            else:
                                retry_failed.append({
                                    'filename': err['filename'],
                                    'cuts_dir': err['cuts_dir'],
                                    'final_dir': err['final_dir'],
                                    'error': '원본 파일 없음'
                                })
                        except Exception as e:
                            retry_failed.append({
                                'filename': err['filename'],
                                'cuts_dir': err['cuts_dir'],
                                'final_dir': err['final_dir'],
                                'error': str(e)
                            })
                    
                    progress.empty()
                    
                    # 결과 업데이트
                    if retry_success > 0:
                        st.success(f"✅ {retry_success}개 재처리 성공!")
                        
                        # 성공한 항목 제거, 실패한 항목만 유지
                        remaining_errors = [err for err in error_cuts 
                                           if err not in selected_errors or err in retry_failed]
                        remaining_errors.extend([e for e in retry_failed if e not in remaining_errors])
                        
                        st.session_state.error_cuts = retry_failed
                        
                        # 결과 통계 업데이트
                        st.session_state.processing_result['success_count'] += retry_success
                        st.session_state.processing_result['error_count'] = len(retry_failed)
                        
                        st.rerun()
                    else:
                        st.error(f"모든 재처리 실패")
                        st.session_state.error_cuts = retry_failed
    
    st.divider()
    
    # 2-6. 원본/결과 비교 및 재처리
    st.markdown("<div class='section-title'>2-6. 원본/결과 비교 및 재처리</div>", unsafe_allow_html=True)
    st.caption("결과가 마음에 들지 않는 컷을 선택하여 재처리할 수 있습니다.")
    
    cut_info_list = st.session_state.cut_info_list
    
    if cut_info_list:
        # 체크박스 버전 (재처리 완료 시 증가)
        checkbox_version = st.session_state.get('reprocess_checkbox_version', 0)
        
        # 페이지 선택 (페이지 번호만 추출해서 표시)
        def get_page_label(info):
            name = info['png_info']['png_stem']
            # page_XXX 패턴 추출
            match = re.search(r'page[_\s]*(\d+)', name, re.IGNORECASE)
            if match:
                return f"page_{match.group(1)}"
            return name.split('_')[-1] if '_' in name else name
        
        page_labels = [get_page_label(info) for info in cut_info_list]
        
        if len(page_labels) > 1:
            selected_page_idx = st.selectbox(
                "페이지 선택", 
                range(len(page_labels)),
                format_func=lambda x: f"{x+1}. {page_labels[x]}",
                key="compare_page_select"
            )
        else:
            selected_page_idx = 0
        
        selected_cut_info = cut_info_list[selected_page_idx]
        cuts_dir = Path(selected_cut_info['cuts_dir'])
        final_dir = Path(selected_cut_info['final_dir'])
        
        cut_files = sorted(cuts_dir.glob("*.png"))
        
        if cut_files:
            # 결과 파일이 있는 컷 목록 (재처리 대상)
            reprocessable_cuts = {}
            for cut_file in cut_files:
                final_file = final_dir / f"{cut_file.stem}_nobubble.png"
                if final_file.exists():
                    reprocessable_cuts[cut_file.name] = {
                        'cut_file': cut_file,
                        'final_file': final_file,
                        'cuts_dir': str(cuts_dir),
                        'final_dir': str(final_dir),
                        'filename': cut_file.name,
                        'png_stem': selected_cut_info['png_info']['png_stem']
                    }
            
            # 선택된 컷 수 계산 (파일명 기반)
            selected_count = sum(1 for fname in reprocessable_cuts.keys() 
                                if st.session_state.get(f"reprocess_{checkbox_version}_{fname}", False))
            
            st.info(f"{page_labels[selected_page_idx]} - {len(cut_files)}개 컷 (재처리 가능: {len(reprocessable_cuts)}개, 선택: {selected_count}개)")
            
            # 컷 선택 슬라이더 (많을 경우)
            if len(cut_files) > 4:
                col_slider, col_count = st.columns([4, 1])
                with col_slider:
                    start_idx = st.slider("표시 시작 위치", 0, len(cut_files) - 1, 0, key="cut_slider")
                with col_count:
                    st.caption(f"{start_idx + 1} ~ {min(start_idx + 4, len(cut_files))} / {len(cut_files)}")
                display_files = cut_files[start_idx:start_idx + 4]
            else:
                display_files = cut_files
            
            # 컷 표시 (원본/결과 나란히 + 체크박스)
            for cut_file in display_files:
                final_file = final_dir / f"{cut_file.stem}_nobubble.png"
                can_reprocess = final_file.exists()
                
                col_check, col_orig, col_result = st.columns([0.5, 2, 2])
                
                with col_check:
                    if can_reprocess:
                        st.checkbox(
                            "재처리",
                            key=f"reprocess_{checkbox_version}_{cut_file.name}",
                            label_visibility="collapsed"
                        )
                    st.caption(cut_file.name)
                
                with col_orig:
                    st.markdown("**원본 (2_cuts)**")
                    if cut_file.exists():
                        # 캐시 방지를 위해 바이트로 읽기
                        with open(cut_file, 'rb') as f:
                            st.image(f.read(), width='stretch')
                
                with col_result:
                    st.markdown("**결과 (3_final)**")
                    if final_file.exists():
                        # 캐시 방지를 위해 바이트로 읽기
                        with open(final_file, 'rb') as f:
                            st.image(f.read(), width='stretch')
                    else:
                        st.warning("결과 파일 없음 (스킵됨)")
                
                st.divider()
            
            # 재처리 버튼
            if reprocessable_cuts:
                # 선택된 컷 수집 (파일명 기반)
                selected_for_reprocess = [
                    item for fname, item in reprocessable_cuts.items() 
                    if st.session_state.get(f"reprocess_{checkbox_version}_{fname}", False)
                ]
                
                # 재처리 설정 (선택된 컷이 있을 때만 표시)
                if selected_for_reprocess:
                    st.markdown("**재처리 설정**")
                    col_model, col_delay = st.columns(2)
                    with col_model:
                        # 마지막 사용 모델을 기본값으로
                        last_model = st.session_state.get('last_model', 'flash')
                        model_options = ["flash", "pro"]
                        default_idx = model_options.index(last_model) if last_model in model_options else 0
                        reprocess_model = st.selectbox(
                            "모델", 
                            model_options, 
                            index=default_idx,
                            format_func=lambda x: "Gemini Flash" if x == "flash" else "Gemini Pro",
                            key="reprocess_model"
                        )
                    with col_delay:
                        reprocess_delay = st.selectbox(
                            "API 간격", 
                            [0, 1, 2, 3], 
                            index=1, 
                            format_func=lambda x: f"{x}초",
                            key="reprocess_delay"
                        )
                    
                    with st.expander("프롬프트 설정", expanded=False):
                        reprocess_prompt = st.text_area(
                            "커스텀 프롬프트", 
                            value=st.session_state.get('last_custom_prompt', DEFAULT_PROMPT), 
                            height=150,
                            key="reprocess_prompt"
                        )
                    
                    st.divider()
                
                col_btn1, col_btn2 = st.columns(2)
                with col_btn1:
                    if st.button(f"🔄 선택 컷 재처리 ({len(selected_for_reprocess)}개)", 
                                type="primary",
                                width='stretch', 
                                disabled=len(selected_for_reprocess) == 0):
                        
                        api_key = st.session_state.get('api_key', '')
                        # 재처리 설정 사용
                        model = st.session_state.get('reprocess_model', 'flash')
                        delay = st.session_state.get('reprocess_delay', 1)
                        prompt = st.session_state.get('reprocess_prompt', DEFAULT_PROMPT)
                        
                        if not api_key:
                            st.error("API 키가 필요합니다.")
                        else:
                            with st.spinner(f"{len(selected_for_reprocess)}개 컷 재처리 중..."):
                                pipeline = WebtoonPipelineGoogle(api_key=api_key, model=model)
                                
                                reprocess_success = 0
                                reprocess_failed = []
                                
                                progress = st.progress(0)
                                status_text = st.empty()
                                
                                for i, item in enumerate(selected_for_reprocess):
                                    progress.progress((i + 1) / len(selected_for_reprocess))
                                    status_text.text(f"처리 중: {item['filename']} ({i+1}/{len(selected_for_reprocess)})")
                                    
                                    try:
                                        input_path = item['cut_file']
                                        output_dir = item['final_dir']
                                        
                                        if input_path.exists():
                                            api_results = pipeline.remove_speech_bubbles(
                                                [str(input_path)],
                                                output_dir,
                                                prompt=prompt,
                                                delay=delay
                                            )
                                            
                                            if api_results and api_results[0].get('success', False):
                                                reprocess_success += 1
                                            else:
                                                error_msg = api_results[0].get('error', 'Unknown') if api_results else 'No result'
                                                reprocess_failed.append(f"{item['filename']}: {error_msg[:50]}")
                                        else:
                                            reprocess_failed.append(f"{item['filename']}: 파일 없음")
                                    except Exception as e:
                                        reprocess_failed.append(f"{item['filename']}: {str(e)[:50]}")
                                
                                progress.empty()
                                status_text.empty()
                                
                                if reprocess_success > 0:
                                    st.success(f"✅ {reprocess_success}개 재처리 완료!")
                                    if reprocess_failed:
                                        st.warning(f"⚠️ {len(reprocess_failed)}개 실패")
                                        for fail_msg in reprocess_failed[:5]:
                                            st.caption(f"  - {fail_msg}")
                                    # 체크박스 버전 증가
                                    st.session_state['reprocess_checkbox_version'] = checkbox_version + 1
                                    st.rerun()
                                else:
                                    st.error(f"재처리 실패")
                                    for fail_msg in reprocess_failed[:5]:
                                        st.caption(f"  - {fail_msg}")
                
                with col_btn2:
                    # 전체 선택 토글 (파일명 기반)
                    all_selected = all(
                        st.session_state.get(f"reprocess_{checkbox_version}_{fname}", False)
                        for fname in reprocessable_cuts.keys()
                    )
                    if all_selected:
                        if st.button("전체 해제", width='stretch'):
                            st.session_state['reprocess_checkbox_version'] = checkbox_version + 1
                            st.rerun()
                    else:
                        if st.button("현재 페이지 전체 선택", width='stretch'):
                            new_version = checkbox_version + 1
                            st.session_state['reprocess_checkbox_version'] = new_version
                            for fname in reprocessable_cuts.keys():
                                st.session_state[f"reprocess_{new_version}_{fname}"] = True
                            st.rerun()
        else:
            st.warning("컷 파일이 없습니다.")
    
    # 새 작업 시작 / 다음 파일 처리 / 컷 보정으로 이동
    col_next, col_new, col_edit = st.columns(3)
    
    with col_next:
        # 처리 완료 후 남은 파일 계산 (현재 선택 파일 + 이미 완료된 파일 제외)
        all_processed = set(st.session_state.get('processed_png_indices', []))
        all_processed.update(st.session_state.selected_png_indices)  # 현재 처리 중인 파일 추가
        
        remaining_indices = [idx for idx in range(len(st.session_state.converted_png_list)) 
                            if idx not in all_processed]
        
        if remaining_indices:
            if st.button(f"처리 완료 → 다음 파일 ({len(remaining_indices)}개 남음)", type="primary", width='stretch'):
                # 현재 처리된 파일 인덱스 저장 (처리 완료로 표시)
                processed_indices = st.session_state.selected_png_indices.copy()
                
                # 선택 초기화 (처리된 파일 제외)
                st.session_state.selected_png_indices = []
                
                # 처리 완료된 파일 목록에 추가
                if 'processed_png_indices' not in st.session_state:
                    st.session_state.processed_png_indices = []
                st.session_state.processed_png_indices.extend(processed_indices)
                
                # 하위 단계 초기화
                st.session_state.cut_split_done = False
                st.session_state.step2_classification_done = False
                st.session_state.processing_done = False
                st.session_state.cut_info_list = []
                st.session_state.cut_classification = {
                    'process': [], 'skip_sfx_only': [], 'skip_no_bubble': [], 
                    'skip_no_text': [], 'skip_bubble_only_cut': []
                }
                st.session_state.processing_result = None
                st.session_state.error_cuts = []
                
                st.rerun()
        else:
            st.success("✅ 모든 파일 처리 완료!")
    
    with col_new:
        if st.button("새 작업 시작", width='stretch'):
            # 상태 초기화
            for key in list(st.session_state.keys()):
                if key != 'file_uploader_key':
                    del st.session_state[key]
            st.session_state.file_uploader_key += 1
            st.rerun()
    with col_edit:
        st.info("👈 사이드바에서 **B. 컷 보정(선택)**을 선택하세요")


st.divider()
st.caption("DOBEDUB v4.1 | Google Gemini API")
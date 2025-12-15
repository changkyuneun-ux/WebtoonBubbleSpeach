"""
웹툰 파이프라인 - 컷 보정 UI
원본 컷과 말풍선 제거 컷을 비교하여 최종 컷을 선택
"""

import streamlit as st
from pathlib import Path
import shutil
from PIL import Image

# 출력 기본 디렉토리
OUTPUT_BASE_DIR = Path.home() / "voicetoon_image"


def get_title_list():
    """작품 목록 조회"""
    if not OUTPUT_BASE_DIR.exists():
        return []
    return sorted([d.name for d in OUTPUT_BASE_DIR.iterdir() if d.is_dir()])


def get_episode_list(title):
    """회차 목록 조회"""
    title_dir = OUTPUT_BASE_DIR / title
    if not title_dir.exists():
        return []
    return sorted([d.name for d in title_dir.iterdir() if d.is_dir()])


def get_source_list(title, episode):
    """원본파일 목록 조회"""
    episode_dir = OUTPUT_BASE_DIR / title / episode
    if not episode_dir.exists():
        return []
    return sorted([d.name for d in episode_dir.iterdir() if d.is_dir()])


def get_page_list(title, episode, source):
    """페이지 목록 조회 (3_final 또는 4_completion이 있는 페이지)"""
    source_dir = OUTPUT_BASE_DIR / title / episode / source
    if not source_dir.exists():
        return []
    
    pages = []
    for d in source_dir.iterdir():
        # 0_source, _temp_png 등 제외, 2_cuts가 있는 디렉토리만 대상
        if d.is_dir() and not d.name.startswith(("0_", "_")):
            cuts_dir = d / "2_cuts"
            final_dir = d / "3_final"
            completion_dir = d / "4_completion"
            
            # 2_cuts가 없으면 페이지 디렉토리가 아님
            if not cuts_dir.exists():
                continue
            
            # cuts 개수
            cuts_count = len(list(cuts_dir.glob("*.png")))
            if cuts_count == 0:
                continue
            
            # final 또는 completion 개수 확인
            final_count = len(list(final_dir.glob("*.png"))) if final_dir.exists() else 0
            completion_count = len(list(completion_dir.glob("*.png"))) if completion_dir.exists() else 0
            
            # 조건: 3_final 또는 4_completion에 파일이 있으면 표시
            if final_count > 0 or completion_count > 0:
                pages.append(d.name)
    
    return sorted(pages)


def get_cuts_data(title, episode, source, page):
    """컷 데이터 조회 (4_completion이 없으면 3_final에서 자동 생성)"""
    page_dir = OUTPUT_BASE_DIR / title / episode / source / page
    cuts_dir = page_dir / "2_cuts"
    finals_dir = page_dir / "3_final"
    completion_dir = page_dir / "4_completion"
    
    cuts_data = []
    
    # 2_cuts의 파일 목록
    if not cuts_dir.exists():
        return []
    
    # 4_completion이 없으면 3_final에서 자동 생성
    if not completion_dir.exists() or not list(completion_dir.glob("*.png")):
        if finals_dir.exists() and list(finals_dir.glob("*.png")):
            completion_dir.mkdir(parents=True, exist_ok=True)
            for final_file in finals_dir.glob("*.png"):
                new_name = final_file.name.replace("_nobubble", "")
                shutil.copy2(final_file, completion_dir / new_name)
    
    for cut_file in sorted(cuts_dir.glob("*.png")):
        cut_name = cut_file.stem  # e.g., page_001_cut_01
        
        # 매칭되는 final 파일 찾기
        final_file = finals_dir / f"{cut_name}_nobubble.png"
        completion_file = completion_dir / f"{cut_name}.png"
        
        # 현재 completion이 원본인지 final인지 확인
        current_source = "final"  # 기본값
        if completion_file.exists() and cut_file.exists():
            # 파일 크기로 비교 (간단한 방법)
            if completion_file.stat().st_size == cut_file.stat().st_size:
                current_source = "cuts"
        
        cuts_data.append({
            'name': cut_name,
            'cuts_path': str(cut_file),
            'final_path': str(final_file) if final_file.exists() else None,
            'completion_path': str(completion_file) if completion_file.exists() else None,
            'current_source': current_source
        })
    
    return cuts_data


def save_completion(title, episode, source, page, selections):
    """선택된 컷을 4_completion에 저장"""
    page_dir = OUTPUT_BASE_DIR / title / episode / source / page
    cuts_dir = page_dir / "2_cuts"
    finals_dir = page_dir / "3_final"
    completion_dir = page_dir / "4_completion"
    
    completion_dir.mkdir(parents=True, exist_ok=True)
    
    saved_count = 0
    for cut_name, source_type in selections.items():
        if source_type == "cuts":
            src_file = cuts_dir / f"{cut_name}.png"
        else:  # final
            src_file = finals_dir / f"{cut_name}_nobubble.png"
        
        if src_file.exists():
            dst_file = completion_dir / f"{cut_name}.png"
            shutil.copy2(src_file, dst_file)
            saved_count += 1
    
    return saved_count


def run_cut_editor_ui():
    """B. 컷 보정(선택) - 3단계 UI 메인 함수"""
    
    st.caption("원본 컷과 말풍선 제거 컷을 비교하여 최종 컷을 선택합니다.")
    
    st.divider()
    
    # Session State 초기화
    if 'cut_editor_selections' not in st.session_state:
        st.session_state.cut_editor_selections = {}
    if 'cut_editor_original' not in st.session_state:
        st.session_state.cut_editor_original = {}
    if 'cut_editor_page_key' not in st.session_state:
        st.session_state.cut_editor_page_key = None
    if 'cut_editor_has_changes' not in st.session_state:
        st.session_state.cut_editor_has_changes = False
    if 'show_save_dialog' not in st.session_state:
        st.session_state.show_save_dialog = False
    if 'pending_page_change' not in st.session_state:
        st.session_state.pending_page_change = None
    
    # 작품/회차/원본파일/페이지 선택
    st.markdown("**페이지 선택**")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        title_list = get_title_list()
        if not title_list:
            st.warning("작품이 없습니다.")
            return
        selected_title = st.selectbox("작품", ["선택..."] + title_list, key="cut_title")
        if selected_title == "선택...":
            selected_title = None
    
    with col2:
        if selected_title:
            episode_list = get_episode_list(selected_title)
            selected_episode = st.selectbox("회차", ["선택..."] + episode_list, key="cut_episode")
            if selected_episode == "선택...":
                selected_episode = None
        else:
            st.selectbox("회차", ["작품을 선택하세요"], disabled=True, key="cut_episode_disabled")
            selected_episode = None
    
    with col3:
        if selected_title and selected_episode:
            source_list = get_source_list(selected_title, selected_episode)
            selected_source = st.selectbox("원본파일", ["선택..."] + source_list, key="cut_source")
            if selected_source == "선택...":
                selected_source = None
        else:
            st.selectbox("원본파일", ["회차를 선택하세요"], disabled=True, key="cut_source_disabled")
            selected_source = None
    
    with col4:
        if selected_title and selected_episode and selected_source:
            page_list = get_page_list(selected_title, selected_episode, selected_source)
            if not page_list:
                st.selectbox("페이지", ["처리된 페이지 없음"], disabled=True, key="cut_page_disabled")
                selected_page = None
            else:
                selected_page = st.selectbox("페이지", ["선택..."] + page_list, key="cut_page")
                if selected_page == "선택...":
                    selected_page = None
        else:
            st.selectbox("페이지", ["원본파일을 선택하세요"], disabled=True, key="cut_page_disabled2")
            selected_page = None
    
    st.divider()
    
    # 페이지 변경 시 저장 확인
    current_page_key = f"{selected_title}/{selected_episode}/{selected_source}/{selected_page}"
    
    if (st.session_state.cut_editor_page_key is not None and 
        st.session_state.cut_editor_page_key != current_page_key and
        st.session_state.cut_editor_has_changes):
        
        st.warning("저장되지 않은 변경사항이 있습니다!")
        col_save, col_discard = st.columns(2)
        with col_save:
            if st.button("저장하고 이동", type="primary", width='stretch'):
                # 이전 페이지 저장
                prev_parts = st.session_state.cut_editor_page_key.split("/")
                if len(prev_parts) == 4:
                    save_completion(prev_parts[0], prev_parts[1], prev_parts[2], prev_parts[3], 
                                  st.session_state.cut_editor_selections)
                    st.success("저장 완료!")
                st.session_state.cut_editor_has_changes = False
                st.session_state.cut_editor_page_key = current_page_key
                st.rerun()
        with col_discard:
            if st.button("저장하지 않고 이동", width='stretch'):
                st.session_state.cut_editor_has_changes = False
                st.session_state.cut_editor_page_key = current_page_key
                st.session_state.cut_editor_selections = {}
                st.session_state.cut_editor_original = {}
                st.rerun()
        return
    
    # 페이지 선택되지 않은 경우
    if not all([selected_title, selected_episode, selected_source, selected_page]):
        st.info("작품, 회차, 원본파일, 페이지를 선택하세요.")
        return
    
    # 페이지 변경 감지 및 데이터 로드
    if st.session_state.cut_editor_page_key != current_page_key:
        st.session_state.cut_editor_page_key = current_page_key
        st.session_state.cut_editor_selections = {}
        st.session_state.cut_editor_original = {}
        st.session_state.cut_editor_has_changes = False
    
    # 컷 데이터 로드
    cuts_data = get_cuts_data(selected_title, selected_episode, selected_source, selected_page)
    
    if not cuts_data:
        st.warning("컷 데이터가 없습니다.")
        return
    
    # 초기 선택 상태 설정
    if not st.session_state.cut_editor_selections:
        for cut in cuts_data:
            st.session_state.cut_editor_selections[cut['name']] = cut['current_source']
            st.session_state.cut_editor_original[cut['name']] = cut['current_source']
    
    # 전체 선택 버튼
    st.markdown("**일괄 선택**")
    col_all_cuts, col_all_finals, col_spacer = st.columns([1, 1, 3])
    with col_all_cuts:
        if st.button("전체 → 원본 컷", width='stretch', key="all_cuts"):
            for cut in cuts_data:
                st.session_state.cut_editor_selections[cut['name']] = "cuts"
            st.session_state.cut_editor_has_changes = True
            st.rerun()
    with col_all_finals:
        if st.button("전체 → 말풍선 제거", width='stretch', key="all_finals"):
            for cut in cuts_data:
                st.session_state.cut_editor_selections[cut['name']] = "final"
            st.session_state.cut_editor_has_changes = True
            st.rerun()
    
    st.divider()
    
    # 3열 레이아웃 헤더
    col_h1, col_h2, col_h3 = st.columns(3)
    with col_h1:
        st.markdown("### 원본 컷 (2_cuts)")
    with col_h2:
        st.markdown("### 말풍선 제거 (3_final)")
    with col_h3:
        st.markdown("### 최종 컷 (4_completion)")
    
    st.divider()
    
    # 컷별 비교 UI
    for idx, cut in enumerate(cuts_data):
        cut_name = cut['name']
        current_selection = st.session_state.cut_editor_selections.get(cut_name, "final")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # 원본 컷
            if cut['cuts_path'] and Path(cut['cuts_path']).exists():
                is_selected = current_selection == "cuts"
                border_style = "3px solid #ff4b4b" if is_selected else "1px solid #ddd"
                
                st.markdown(f"<div style='border: {border_style}; border-radius: 5px; padding: 5px;'>", 
                           unsafe_allow_html=True)
                st.image(cut['cuts_path'], width='stretch')
                
                if st.button(f"{'선택됨' if is_selected else '○ 선택'}", 
                           key=f"sel_cuts_{idx}",
                           type="primary" if is_selected else "secondary"):
                    st.session_state.cut_editor_selections[cut_name] = "cuts"
                    st.session_state.cut_editor_has_changes = True
                    st.rerun()
                st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.warning("원본 컷 없음")
        
        with col2:
            # 말풍선 제거 컷
            if cut['final_path'] and Path(cut['final_path']).exists():
                is_selected = current_selection == "final"
                border_style = "3px solid #ff4b4b" if is_selected else "1px solid #ddd"
                
                st.markdown(f"<div style='border: {border_style}; border-radius: 5px; padding: 5px;'>", 
                           unsafe_allow_html=True)
                st.image(cut['final_path'], width='stretch')
                
                if st.button(f"{'선택됨' if is_selected else '○ 선택'}", 
                           key=f"sel_final_{idx}",
                           type="primary" if is_selected else "secondary"):
                    st.session_state.cut_editor_selections[cut_name] = "final"
                    st.session_state.cut_editor_has_changes = True
                    st.rerun()
                st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.warning("말풍선 제거 컷 없음")
        
        with col3:
            # 현재 최종 컷 미리보기
            if current_selection == "cuts" and cut['cuts_path']:
                preview_path = cut['cuts_path']
                source_label = "🔵 원본 컷"
            elif cut['final_path']:
                preview_path = cut['final_path']
                source_label = "🟢 말풍선 제거"
            else:
                preview_path = None
                source_label = ""
            
            if preview_path and Path(preview_path).exists():
                st.image(preview_path, width='stretch')
                st.caption(f"현재 선택: {source_label}")
            else:
                st.warning("미리보기 없음")
        
        st.divider()
    
    # 변경사항 확인
    has_changes = False
    for cut_name, selection in st.session_state.cut_editor_selections.items():
        if st.session_state.cut_editor_original.get(cut_name) != selection:
            has_changes = True
            break
    st.session_state.cut_editor_has_changes = has_changes
    
    # 저장 버튼
    st.markdown("---")
    col_status, col_save = st.columns([3, 1])
    
    with col_status:
        if has_changes:
            changed_count = sum(1 for k, v in st.session_state.cut_editor_selections.items() 
                              if st.session_state.cut_editor_original.get(k) != v)
            st.warning(f"{changed_count}개 컷의 변경사항이 있습니다.")
        else:
            st.info("변경사항 없음")
    
    with col_save:
        if st.button("저장", type="primary", width='stretch', disabled=not has_changes):
            saved = save_completion(selected_title, selected_episode, selected_source, selected_page,
                                   st.session_state.cut_editor_selections)
            st.success(f"완료: {saved}개 파일 저장 완료!")
            
            # 원본 상태 업데이트
            st.session_state.cut_editor_original = st.session_state.cut_editor_selections.copy()
            st.session_state.cut_editor_has_changes = False
            st.rerun()
    
    # 일괄 다운로드 섹션
    st.divider()
    st.markdown("### 일괄 다운로드")
    
    # 다운로드 범위 선택
    download_scope = st.radio(
        "다운로드 범위",
        ["원본 컷 (2_cuts)", "말풍선 제거 (3_final)", "최종 컷 (4_completion)"],
        horizontal=True,
        key="download_scope"
    )
    
    # 다운로드 대상 디렉토리 결정
    target_dir = OUTPUT_BASE_DIR / selected_title / selected_episode / selected_source / selected_page
    
    if download_scope == "원본 컷 (2_cuts)":
        source_dir = target_dir / "2_cuts"
        folder_name = "2_cuts"
    elif download_scope == "말풍선 제거 (3_final)":
        source_dir = target_dir / "3_final"
        folder_name = "3_final"
    else:  # 최종 컷 (4_completion)
        source_dir = target_dir / "4_completion"
        folder_name = "4_completion"
    
    col_dl_info, col_dl_btn = st.columns([3, 1])
    
    with col_dl_info:
        if source_dir.exists():
            file_count = len(list(source_dir.glob("*.png")))
            st.info(f"{selected_page}/{folder_name} - {file_count}개 파일")
        else:
            file_count = 0
            st.warning(f"{folder_name} 폴더가 없습니다")
    
    with col_dl_btn:
        # ZIP 파일 미리 생성
        import zipfile
        import io
        
        zip_buffer = io.BytesIO()
        zip_name = f"{selected_title}_{selected_episode}_{selected_page}_{folder_name}.zip"
        
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
            if source_dir.exists():
                for png_file in sorted(source_dir.glob("*.png")):
                    arcname = f"{selected_page}/{png_file.name}"
                    zf.write(png_file, arcname)
        
        zip_buffer.seek(0)
        
        st.download_button(
            label="ZIP 다운로드",
            data=zip_buffer.getvalue(),
            file_name=zip_name,
            mime="application/zip",
            width='stretch',
            disabled=file_count == 0
        )


# 독립 실행용
if __name__ == "__main__":
    st.set_page_config(
        page_title="웹툰 파이프라인 - 컷 보정",
        page_icon="🖼️",
        layout="wide"
    )
    
    st.title("3단계: 웹툰 컷 조정")
    st.info("※ 2단계까지가 정규 처리 단계입니다. 이 단계는 선택적으로 사용합니다.")
    run_cut_editor_ui()
    
    st.divider()
    st.caption("DOBEDUB v4.1 | 컷 보정")
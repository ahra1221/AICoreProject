"""Create honest checklists from the supplied template; never invent peer reviews."""
from __future__ import annotations
import csv
import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from xml.dom import minidom
from zipfile import ZipFile, ZIP_DEFLATED

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
EVIDENCE = DOCS / "evidence"
ROWS = [
    (1, "F2001", "통과", "local-index.json", "실제 로컬 Chroma 964건=964청크, 속성 확인. 테스트 벡터 사용; 의미 품질 제외"),
    (2, "F2002", "통과", "local-index.json", "실제 두 번 색인: 964→964, 두 번째 변경 0건"),
    (3, "F2003/F2004", "미확인", "offline-flow.json; wikipedia-api.json", "모의 P1/P2 및 실 API 수집 검증 완료. 실제 LLM P1/P2는 키·예산 미설정으로 미실행"),
    (4, "F2004/NF2002", "미확인", "offline-flow.json; wikipedia-api.json", "모의 동일 질문의 P1/P2 재호출 없음·실 API 반복 HTTP 0회 확인. 실제 LLM 질문 화면 미실행"),
    (5, "F2005", "통과", "local-index.json", "실제 로컬 재색인: chunk/overlap 120/20의 12청크 → 400/40의 4청크"),
    (6, "F2006", "통과", "offline-flow.json", "동일 질문·실제 로컬 검색의 Top-k 1/3: 외부 1/3건(메타데이터 1건 별도). 테스트 벡터 사용"),
    (7, "F2007", "통과", "graph.mmd; offline-flow.json", "실행 노드 retrieve/generate 확인. START/END는 경계"),
    (8, "F2008", "미확인", "offline-tests.txt; offline-flow.json", "번호·출처 매핑 자동 검증 완료. 실제 생성 답변의 화면 대조 미실행"),
    (9, "F2007/F2008", "미확인", "offline-tests.txt", "무근거·잘못된 인용 차단 확인. 실제 LLM의 의미상 환각 억제는 미검증"),
    (10, "F2009", "미확인", "offline-tests.txt", "movie_id별 대화 분리 함수 통과. 두 영화의 실제 질문 UI 전환 미실행"),
    (11, "F2011", "미확인", "offline-tests.txt", "실제 964편으로 연도·장르·감독 필터 통과. 실제 임베딩 조건탐색 UI 미실행"),
    (12, "F2010", "미확인", "../rag-design.md", "사실검증 프롬프트·근거 경로 유지. 실제 모델 사실검증 미실행"),
    (13, "F2012", "미확인", "ui-guidance.json", "키 없음·색인 없음 UI 자동 실행 확인. 실제 질문 진행 화면은 미확인"),
    (14, "F2012", "해당없음", "../requirements.md", "일반 영화 도메인으로 건강·금융 판단 기능 없음"),
    (15, "F2013", "해당없음", "../requirements.md", "별도 HTML/CSS/JS UI 심화 선택하지 않음"),
    (16, "F2014", "해당없음", "../requirements.md", "MMR 심화 선택하지 않음"),
    (17, "F2015", "해당없음", "../requirements.md", "RAG 유무 비교 심화 선택하지 않음"),
    (18, "F2015", "해당없음", "../requirements.md", "RAG 유무 비교 심화 선택하지 않음"),
    (19, "F2016", "미확인", "offline-flow.json", "TMDB/Wikipedia 종류 보존 확인. 실제 답변 UI 최종 점검 미실행"),
    (20, "NF2001", "미확인", "offline-flow.json", "모의 첫/두 번째 시간만 기록. 실제 LLM 서비스 시간 및 사용성 미확인; 임의 제한시간 없음"),
    (21, "NF2003/NF2004", "통과", "ui-safe-error.json; security-scan.json", "고의 예외 UI 실행에서 상세정보 비노출, 소스/문서 정적 키 검사. 학생의 최종 환경 점검은 별도"),
    (22, "NF2005", "통과", "../rag-design.md", "수집·색인·검색·생성·UI 모듈 분리 확인"),
    (23, "NF2006", "통과", "offline-tests.txt; wikipedia-api.json", "직렬 간격/헤더/재시도/maxlag/Retry-After 코드·모의 검증 및 실 API 성공"),
    (24, "NF2007", "통과", "../../README.md", "TMDB/Wikipedia 출처·이용조건·원문/판본 표시 기록"),
    (25, "필수", "미확인", "../test-results.md", "유료 모델 실제 실행과 학생 직접 점검 미완료로 전체 필수 합격 주장하지 않음"),
    (26, "필수", "미확인", "../submission.md", "초기 안내 화면 캡처만 가능. 실제 질문·사실검증 등 모든 필수 캡처는 미완료"),
    (27, "필수", "통과", "../submission.md", "요구사항 정의·설계·소스·README·대조표·결과표·체크리스트 포함. 상호 평가는 미실시로 표시"),
]


def text_cell(doc, root, address, value):
    cells = {node.getAttribute("r"): node for node in root.getElementsByTagName("c")}
    cell = cells[address]
    for child in list(cell.childNodes):
        cell.removeChild(child)
    cell.setAttribute("t", "inlineStr")
    inline, text = doc.createElement("is"), doc.createElement("t")
    text.setAttribute("xml:space", "preserve")
    text.appendChild(doc.createTextNode(str(value)))
    inline.appendChild(text)
    cell.appendChild(inline)


def checklist(template, destination, peer=False):
    with ZipFile(template) as original:
        xml = minidom.parseString(original.read("xl/worksheets/sheet1.xml"))
        root = xml.documentElement
        text_cell(xml, root, "C3", "상호 (미실시)" if peer else "셀프 (자동 사전점검)")
        text_cell(xml, root, "C4", "원본 저장소: ahra1221/AICoreProject")
        text_cell(xml, root, "C5", "실제 교육생 점검자 기입 필요" if peer else "Codex 자동 검증 / 학생 직접 점검 별도")
        text_cell(xml, root, "C6", "미실시" if peer else datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"))
        counts = Counter()
        for number, requirement, status, evidence, note in ROWS:
            row = number + 8
            if peer and status != "해당없음":
                status, evidence, note = "미확인", "", "학생 상호 평가 미실시. 점검자가 직접 확인 후 기입할 것"
            counts[status] += 1
            for col, value in zip("GHI", (status, evidence, note)):
                text_cell(xml, root, f"{col}{row}", value)
        cells = {node.getAttribute("r"): node for node in root.getElementsByTagName("c")}
        for row, label in ((37, "통과"), (38, "실패"), (39, "미확인"), (40, "해당없음"), (41, "미기입"), (42, "전체")):
            values = cells[f"F{row}"].getElementsByTagName("v")
            if values:
                for child in list(values[0].childNodes):
                    values[0].removeChild(child)
                values[0].appendChild(xml.createTextNode(str(27 if label == "전체" else counts[label])))
        with ZipFile(destination, "w", ZIP_DEFLATED) as output:
            for item in original.infolist():
                output.writestr(item, xml.toxml(encoding="utf-8") if item.filename == "xl/worksheets/sheet1.xml" else original.read(item.filename))
    return dict(counts)


def security_scan():
    patterns = [re.compile(r"\bsk-[A-Za-z0-9_-]{24,}"), re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}"),
                re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----")]
    files = [p for folder in ("src", "scripts", "tests", "docs") for p in (ROOT / folder).rglob("*")
             if p.suffix in {".py", ".md", ".csv", ".json", ".txt"} and "__pycache__" not in p.parts]
    files += [ROOT / "app.py", ROOT / "README.md", ROOT / ".env.example"]
    matches = [str(p.relative_to(ROOT)) for p in files if any(pat.search(p.read_text(encoding="utf-8")) for pat in patterns)]
    empty_example = all(not line.partition("=")[2].strip() for line in (ROOT / ".env.example").read_text().splitlines() if "=" in line)
    report = {"files_scanned": len(files), "matching_paths": matches, "env_example_values_empty": empty_example,
              "scope": "source/docs known key formats; does not guarantee detection of every secret format"}
    (EVIDENCE / "security-scan.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    assert not matches and empty_example


def main():
    security_scan()
    counts = checklist(DOCS / "templates/checklist-original.xlsx", DOCS / "checklist-self.xlsx")
    checklist(DOCS / "templates/checklist-original.xlsx", DOCS / "checklist-peer.xlsx", peer=True)
    with (DOCS / "requirements-matrix.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["체크리스트 번호", "요구사항", "자동 사전점검 결과", "증빙 (evidence/ 기준)", "범위와 남은 점검"])
        writer.writerows(ROWS)
    result = json.loads((EVIDENCE / "offline-tests.json").read_text())
    flow = json.loads((EVIDENCE / "offline-flow.json").read_text(encoding="utf-8"))
    api = json.loads((EVIDENCE / "wikipedia-api.json").read_text(encoding="utf-8"))
    report = f"""# 테스트 결과표

기록 시점: {datetime.now(timezone.utc).isoformat()}

| 검증 | 관측 결과 | 한계 |
|---|---|---|
| 네트워크 차단 자동 테스트 | {result['tests']}개, 실패 {result['failures']}, 오류 {result['errors']} | 모델/HTTP는 모의 객체, Chroma와 Streamlit 테스트는 실제 로컬 실행 |
| F2001/F2002 | 실제 Chroma 964건, 재색인 964건, 두 번째 변경 0건 | 테스트용 해시 벡터, 실제 임베딩 품질 아님 |
| F2005 | chunk/overlap 120/20: 12개 → 400/40: 4개 | 같은 테스트 본문을 실제 재색인 |
| F2006 | Top-k 1: 메타데이터 포함 2건 / Top-k 3: 4건 | 실제 로컬 검색, 테스트 벡터 |
| F2007 | {flow['graph_nodes']} | 실제 LangGraph 실행, 모델 응답은 모의 |
| P1/P2 캐시 | 첫 실행 P1/P2 각 1회, 반복 실행 추가 0회 | 실제 LLM 요청 없음 |
| 첫 질문 / 두 번째 질문 시간 | {flow['runs'][0]['seconds']:.4f}초 / {flow['runs'][1]['seconds']:.4f}초 | 모의 모델/로컬 데이터. NF2001 실서비스 사용성은 미확인 |
| 실제 Wikipedia | 성공={api['success']}, HTTP {api['http_requests']}회, 본문 {api.get('body_characters', 0)}자 | 무료 공개 API, P1/P2 모델 미실행 |
| 실 API 반복 | 추가 HTTP {api.get('repeat_added_http_requests', '미확인')}회 | API 응답 캐시 검증 |
| UI 키/색인 없음 | Streamlit AppTest 실제 실행 통과 | 실제 질문 진행 및 전체 화면 캡처 별도 |
| UI 고의 오류 | 일반 오류만 표시, 예외 원문/추적 비노출 | AppTest로 관측; 원문은 화면 외 로컬 로깅 가능 |
| 실제 GMS·최종 답변 품질 | 미실행 | 별도 키와 호출 예산 미승인 |
| 학생 상호 평가 | 미실시 | 타인 점검·평가를 작성하지 않음 |

셀프 파일은 자동 사전점검 결과이며 학생의 직접 검증을 대체하지 않습니다. 집계: {json.dumps(counts, ensure_ascii=False)}. 모든 미확인 항목에 사유를 적었습니다. 최초 개발 중 발견한 캐시 시각 불일치와 Windows 임시 경로 문제를 수정한 뒤 위 결과를 다시 확인했습니다.
"""
    (DOCS / "test-results.md").write_text(report, encoding="utf-8")
    print(json.dumps({"checklist_counts": counts, "tests": result}, ensure_ascii=False))


if __name__ == "__main__":
    main()

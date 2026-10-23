# 제출물 안내 및 남은 점검

이 폴더는 수정 작업의 검토 묶음입니다. 전체 과제의 최종 합격을 의미하지 않습니다.

| 항목 | 위치 |
|---|---|
| 프로젝트 요구사항 정의 | `requirements.md` (전달된 요약과 첨부 기준을 영화 도메인에 적용) |
| RAG 파이프라인 설계 | `rag-design.md`, `evidence/graph.mmd` |
| 소스 및 실행 방법 | 상위 `src/`, `app.py`, `scripts/`, `tests/`, `README.md` |
| 요구사항 대조표 | `requirements-matrix.csv` (27개 항목) |
| 테스트 결과표 | `test-results.md`, `evidence/offline-tests.json`, `evidence/offline-tests.txt` |
| 셀프 체크리스트 | `checklist-self.xlsx` (자동 사전점검 표시, 확인하지 않은 항목 미확인) |
| 상호 체크리스트 | `checklist-peer.xlsx` (상호 점검 미실시, 점검자/피드백 날조 없음) |
| 사용자 첨부 원본 | `templates/checklist-original.pdf`, `templates/checklist-original.xlsx` |
| 실 API 증빙 | `evidence/wikipedia-api.json`, `evidence/wikipedia-king-kong.json` |
| 색인·캐시·Top-k 증빙 | `evidence/local-index.json`, `evidence/offline-flow.json` |
| UI 자동 검증 | `evidence/ui-guidance.json`, `evidence/ui-safe-error.json` |

원본 XLSX의 확인 항목·확인 방법·스타일·수식은 유지하고 입력 칸과 집계 캐시만 채웠습니다. 작성자/점검자에 다른 사람을 사칭하지 않았습니다. 원본 PDF는 변경하지 않았습니다.

추가로 필요한 실제 점검:

1. 이 프로젝트에서 사용할 GMS 키와 모델/임베딩 호출 예산을 별도로 준비합니다.
2. README의 메타데이터 색인 방법으로 최소 10편을 준비한 뒤 P1/P2와 실 API가 연결된 질문을 제출합니다.
3. 같은 질문을 두 번, 동명 영화 관련 질문, 문서에 없는 질문, 과학·역사 사실 검증을 실행하고 화면과 첫/두 번째 시간을 기록합니다. 테스트 시간을 실제 서비스 시간으로 사용하지 않습니다.
4. 두 영화 사이 대화 전환, 조건 탐색, 진행 중 화면과 근거 번호·출처 종류를 직접 확인하고 캡처합니다.
5. 실제 교육생이 셀프/상호 점검을 진행하고 남은 미확인을 갱신합니다. 학생 상호 평가는 자동으로 작성할 수 없습니다.

선택하지 않은 HTML/CSS/JS UI, MMR, 5질문 RAG 유무 비교는 해당없음입니다. 최종 키·로그·DB·가상환경은 제출 ZIP에서 제외합니다. 제공된 영화/위키 데이터는 출처 설명과 함께 포함합니다.

## 브라우저 화면 증빙

- `evidence/13_empty_index.jpg`: 실제 8521 초기 화면, 키/색인 없음 안내.
- `evidence/13_missing_key.jpg`: 조건 탐색 요청 후 키 없음 안내, 모델 호출 없음.
- 전체 질문/진행/최종 답변 캡처는 별도 키·예산이 없어 미완료입니다.

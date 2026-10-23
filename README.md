# Movie Lab

Movie Lab은 영화 메타데이터와 한국어 위키백과 자료를 검색해 근거와 함께 답변하는 Streamlit 기반 영화 지식 서비스입니다. 영화별 Q&A, 자연어 조건 영화 탐색, 영화 속 과학·역사적 사실 검증 화면을 제공합니다.

## 프로젝트 구조

```text
.
├── app.py                         # Streamlit UI
├── requirements.txt               # 검증된 Python 의존성
├── .env                            # 직접 생성하는 로컬 환경 설정 (Git 제외)
├── data/
│   ├── movies.json                # 영화 964편 메타데이터
│   └── movielab_wiki/             # 데이터 설명, manifest, 위키 문서 스냅샷
├── src/
│   ├── config.py                  # 환경 설정과 경로
│   ├── data_loader.py             # 영화·위키 JSON 로더
│   ├── indexer.py                 # 청킹과 ChromaDB 색인
│   ├── collector.py               # 검색어·문서 선별, Wikipedia API fallback
│   ├── retriever.py               # 영화 범위 근거 검색과 인용 연결
│   ├── prompts.py                 # JSON 검색 계획·근거 답변 프롬프트
│   ├── graph.py                   # LangGraph retrieve/generate 흐름
│   └── services.py                # 자연어 영화 조건 탐색
├── scripts/build_index.py         # 색인 생성 및 증분 갱신
├── tests/                         # 외부 API 없는 단위 테스트
└── chroma_db/                     # 로컬 벡터 데이터베이스
```

## 요구 사항

- Python 3.10 이상 (개발 및 검증: Python 3.11)
- GMS API 키
- 최초 색인 작업에 사용할 임베딩 모델 접근 권한

## 설치 및 환경 설정

PowerShell에서 가상 환경을 만들고 의존성을 설치합니다.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
New-Item .env -ItemType File
```

`.env`를 열어 다음 변수를 직접 입력하세요. `.env`는 Git에 포함되지 않습니다.

```dotenv
GMS_KEY=your_gms_api_key
WIKI_CONTACT=your-email@example.com
GMS_MODEL=gpt-5.4-nano
GMS_EMBEDDING_MODEL=text-embedding-3-small
```

`GMS_MODEL`과 `GMS_EMBEDDING_MODEL`은 GMS 계정에서 사용할 수 있는 모델에 맞게 조정하세요. 기본 모델명은 실습 자료와 연결 확인에 사용한 예시입니다. Wikipedia API fallback을 쓰려면 `WIKI_CONTACT`에 연락 가능한 이메일 주소를 입력하세요.

## 색인 및 실행

제공된 데이터는 `data/` 아래에 포함되어 있습니다. 최초 색인에는 임베딩 API 호출이 필요하며, ChromaDB 데이터는 `chroma_db/`에 저장되어 재실행 시 변경된 문서만 임베딩합니다.

```powershell
# 10편 / 위키 청크 50개 한도 테스트. 전체 색인은 실행하지 않습니다.
python scripts/build_index.py --test-mode

# 테스트가 성공하면 전체 데이터 색인은 별도로 직접 실행합니다.
python scripts/build_index.py
streamlit run app.py
```

테스트 모드는 영화 5~100편과 위키 청크 20~50개만 색인하며, 기본값은 각각 100편과 50개입니다. 실행 전 전체 처리량의 상한을 알리고 임베딩 연결 및 모델을 한 건의 사전 요청으로 확인합니다. 성공 여부와 실제 색인된 수를 출력합니다. 테스트 데이터는 기본 `chroma_db/`에 추가되어 앱에서 확인할 수 있으며, 색인 완료 후 전체 색인은 별도 명령으로 직접 실행해야 합니다.

전체 색인 규모만 확인하려면 `python scripts/build_index.py --estimate-only`를 실행하세요. API 요청이나 데이터 기록은 하지 않습니다. 색인 스크립트는 `movie_meta`와 `external_docs` 두 컬렉션을 생성합니다. 메타데이터는 영화 ID로 upsert하고, 위키 문서 청크는 `movie_id:page_id:chunk_index` ID를 사용해 재실행 시 중복을 방지합니다. 실패로 중단된 경우 다시 실행하면 이미 색인된 동일 항목은 건너뛰고 누락되거나 변경된 항목만 처리합니다.

## 테스트

네트워크·LLM 호출 없이 데이터 로딩, 청킹, ID 안정성, 검색 범위와 인용 연결을 점검합니다.

```powershell
python -m unittest discover -s tests -v
```

## 데이터 출처 및 이용 조건

위키 문서 스냅샷은 한국어 위키백과 자료를 기반으로 하며, 배포 설명에 명시된 **CC BY-SA 4.0** 조건을 따릅니다. 화면에 문서 제목과 원문 링크를 표시하고, 배포 시에도 출처 및 라이선스 조건을 유지하세요. 제공 데이터에 대한 자세한 내용은 [data/movielab_wiki/README.md](data/movielab_wiki/README.md)를 참고하세요. 영화 메타데이터의 TMDB 출처·이용 조건은 제공된 데이터 자료를 확인하세요.

## 제한 사항

- 실제 답변 및 임베딩 호출은 설정된 GMS 계정과 모델 지원 여부에 따라 달라집니다.
- 외부 Wikipedia API는 선택된 영화의 사전 수집 자료가 없을 때만 연락처 설정 후 사용합니다. 검색 응답은 `data/.cache/`에 캐시됩니다.
- `chroma_db/`와 `.env`는 로컬 전용이며 저장소에 커밋하지 마세요.

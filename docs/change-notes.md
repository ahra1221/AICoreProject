# 변경·검증·원격 반영 상태

기준: 51518f675c99c6cf5f4d817874f1658ff91c47ba, 원격 origin=https://github.com/ahra1221/AICoreProject.git, 브랜치 master.

공식 Wikipedia 검색→P2 선별→본문 수집, 영화 제목/연도 구분, 질문 캐시, 출처/판본·처리 시간 표시를 추가했습니다. 원본 retrieve/generate 두 노드와 영화 Q&A/조건 탐색/사실 검증을 유지했습니다. 원본 데이터·기존 DB 3,731개 파일 해시가 동일함을 확인했습니다.

37개 오프라인 테스트와 실제 Wikipedia의 인터스텔라/킹콩(2005) 수집이 성공했습니다. 실제 GMS 모델/임베딩·전체 답변 품질과 학생 상호 평가는 미실행입니다. 상세 범위는 test-results.md와 체크리스트에 적었습니다. 프로젝트 전용 가상환경 의존성 검사와 git diff --check가 통과했습니다.

사용자는 후속으로 NIGHTPURI 계정에서 원본 저장소에 커밋·push를 승인했습니다. Windows Git Credential Manager의 기존 인증을 GitHub /user로 확인한 결과 NIGHTPURI가 맞습니다. 동일 인증으로 조회한 ahra1221/AICoreProject의 permissions.push는 false입니다. 따라서 원격 push 명령을 실행하지 않았고, fork·remote 변경·계정 변경·재시도도 하지 않습니다. 승인된 로컬 커밋만 작성합니다. 서버 측 쓰기 권한을 해결해야 업로드할 수 있습니다.

작성자 정보는 기존 NIGHTPURI와 GitHub noreply 주소 그대로이며 전역/로컬 Git 설정을 변경하지 않습니다. 로컬 커밋 SHA와 검토 ZIP의 메타데이터는 submission/manifest.json에 남깁니다. 이 폴더는 Git에서 제외됩니다.

# Windows 로컬 번역판

일본어 만화를 한국어로 번역하는 개인용 구성입니다. 기본 구성은 CTD(대사 영역 검출), manga_ocr(일본어 인식), LaMa(원문 제거), Ollama의 Qwen3.5 9B Q4_K_M(한국어 번역)입니다. 모델마다 선택하는 복잡한 설치 절차 대신 준비 화면에서 필요한 구성만 설치합니다. 실제 만화 번역의 최상위 품질이나 모든 PC의 처리 속도를 보장하는 비교 검증은 아직 완료되지 않았습니다.

## 설치 및 실행

1. 이 저장소의 소스 또는 GitHub Actions의 `BallonsTranslator-Local-Windows-source` 파일을 다운로드하고, 쓰기 가능한 폴더에 압축을 풉니다. GitHub의 ZIP 안에 실행 ZIP이 있으면 안쪽 ZIP도 풀어야 합니다.
2. `launch_local.bat`를 실행합니다. 최초 실행은 인터넷이 필요하며, Python과 기존 앱·선택 모델의 필수 패키지를 설치합니다.
3. Ollama가 없으면 공식 설치 파일의 Windows 서명과 게시자를 확인한 뒤 설치 화면을 엽니다. 설치 화면을 완료합니다.
4. 프로그램의 준비 화면에서 이미지 처리 모델과 번역 모델을 각각 처음 한 번 다운로드합니다. 실행 환경 설치만 끝난 상태에서는 아직 번역을 시작할 수 없습니다. 번역 모델 태그는 `qwen3.5:9b-q4_K_M`이며 모델 자체의 다운로드는 약 6.6GB입니다. 전체 설치 공간은 Python, PyTorch, OCR·복원 모델, Ollama의 추가 공간이 필요합니다.
5. 필요한 항목의 준비 상태를 확인한 뒤 폴더를 열고 번역합니다.

Windows 10 22H2 이상, 64비트 x86 PC를 대상으로 합니다. Ollama의 GPU 지원과 요구 드라이버는 [공식 Windows 문서](https://docs.ollama.com/windows)를 따릅니다. 이미지 처리용 PyTorch는 기존 프로그램의 GPU 감지기로 설치 프로필을 선택합니다. 기본 로컬판은 이미지 검출·OCR·복원을 먼저 진행하고 해당 모델을 메모리에서 해제한 뒤 Ollama 번역을 진행하는 저메모리 경로를 사용합니다. 따라서 전체 페이지의 이미지 처리 진행률이 먼저 올라가고 번역이 나중에 시작되는 것은 정상입니다. 이 방식도 모델 자체의 메모리 요구를 없애지는 않으므로, RAM·VRAM 여유가 부족하면 느려지거나 모델 로딩이 실패할 수 있습니다.

수동으로 실행 환경만 준비할 때는 저장소 폴더의 PowerShell에서 다음 명령을 사용합니다.

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\install_local.ps1 -InstallOllama
```

## 저장 위치와 로컬 처리

- Python 실행 환경: 프로그램 폴더의 `local_runtime_win`
- 앱의 이미지 처리 모델: `data/models`
- 로컬판 설정: `config/hebi-local.json`
- Ollama 모델: 기본적으로 사용자 프로필의 `.ollama/models`; 이미 설정한 `OLLAMA_MODELS`가 있으면 해당 경로
- 실행 로그: 프로그램 폴더의 `logs`

기존 Python 환경, 모델, 설정을 자동으로 삭제하지 않습니다. 설치 실패 시 같은 명령으로 재시도할 수 있습니다. Python 실행 파일 없이 `local_runtime_win` 폴더만 남아 있으면 덮어쓰지 않고 안내합니다. 폴더를 백업하거나 다른 이름으로 옮긴 뒤 다시 시도하십시오.

실행기는 이 프로그램 전용 Ollama 서버를 `127.0.0.1:11435`에서 시작하고, 해당 자식 프로세스에 `OLLAMA_NO_CLOUD=1`을 적용합니다. 기존 Ollama 서버나 Windows의 사용자 환경 변수를 변경하지 않습니다. 앱을 정상 종료하면 실행기가 자신이 시작한 서버만 종료합니다. 같은 포트를 이미 사용 중이면 기존 프로세스를 건드리지 않고 실행을 중단합니다. 강제 종료로 전용 서버가 남았을 때는 작업 관리자에서 해당 프로세스를 확인해야 합니다.

Hugging Face와 Transformers는 실행기에서 오프라인 로딩 설정을 사용하므로, 설치한 OCR을 불러올 때 누락된 파일을 몰래 온라인에서 가져오지 않습니다. 앱 준비 화면의 명시적인 모델 다운로드에는 인터넷을 사용합니다. 최초 준비와 업데이트 이후 네트워크를 차단한 실제 Windows PC에서 전체 번역을 완료하는 최종 검증은 별도로 필요합니다.

## 배포 형식과 검증

현재 배포 파일은 **소스와 설치·실행 스크립트가 포함된 ZIP**입니다. Python·Ollama·모델이 모두 포함된 단독 EXE나 오프라인 완성 배포본은 아닙니다. GitHub 비공개 저장소를 다시 복제하지 않으므로 다운로드한 소스가 있으면 실행 준비를 진행할 수 있습니다.

개발자가 패키지를 만들 때는 추가할 소스를 Git에 먼저 등록하고 다음 명령을 실행합니다. Git에 등록된 파일의 현재 작업 내용을 사용하며, 등록하지 않은 개인 파일, 런타임, 모델, 개인 설정, 일반적인 인증 파일은 제외합니다. 기존 ZIP을 덮어쓰지 않습니다.

```powershell
python scripts/build_local_package.py --output release/BallonsTranslator-Local-Windows-source.zip
```

GitHub Actions의 **Local Windows bootstrap package** 작업은 Windows에서 PowerShell 구문·환경·실패 전파 검사, Python 컴파일, 로컬판 관련 회귀 테스트와 ZIP 생성을 수행합니다. 회귀 검사는 Qt의 전역 객체가 서로 다른 테스트 파일에 영향을 주지 않도록 파일마다 새 Python 프로세스에서 실행하며, 실패·충돌·시간 초과가 발생하면 패키지를 만들지 않습니다. 테스트를 통과해야 다운로드용 artifact가 생성됩니다. 모델 다운로드나 GPU 추론, 실제 만화 번역은 CI에서 수행하지 않습니다. GitHub Actions가 아직 실행되지 않았거나 실패한 경우 Windows 검증이 완료된 것으로 볼 수 없습니다.

CI의 회귀 범위는 `scripts/run_local_regressions.py`에 명시되어 있습니다. 로컬 설치 상태, Ollama 요청, 작업 화면, 이어서 처리·저메모리 경로, 번역 응답 형식, 파이프라인 오류, 실행 창, 이미지 저장·읽기, 재실행과 의존성 선택을 검사합니다. 기존 전체 테스트를 모두 실행하는 검증은 아니며, 원본의 문맥 기본값 불일치(테스트는 4,096, 현재 기본값은 10,000) 같은 별도 이슈는 이 검증으로 해결되었다고 볼 수 없습니다. CI의 `openai` 패키지는 기존 공용 LLM 통신 회귀 테스트용이며 로컬판 설치 요구사항에는 추가하지 않습니다.

원본 GPL 라이선스와 소스는 패키지에 포함됩니다. 모델 이용 조건은 각 배포처를 따릅니다.

공식 참고: [Python 3.12.10](https://www.python.org/downloads/release/python-31210/), [Ollama Windows](https://docs.ollama.com/windows), [Ollama 로컬 전용 설정](https://docs.ollama.com/faq#how-do-i-disable-ollama-s-cloud-features).

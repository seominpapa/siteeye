# sitesee

건설 현장 CCTV 영상에서 **최근 5초의 행동 흐름**을 뽑아 TypeSafe AI의 의사결정 모델 **Jev 1.14**에 보내고, 협착·추락·실신 같은 **사고 전조(near-miss)** 를 미리 판단해 조치를 안내하는 파이프라인과 로컬 웹앱입니다. (코드 패키지 이름은 `siteeye`, 화면에는 "SiteEye"로 표시됩니다.)

| 데모 ② 굴착기 접근 + 작업자 등 돌림 | 데모 ④ 역광으로 안전모 인식 저하 → 이미지 폴백 |
|---|---|
| ![demo 2](docs/images/demo-scenario2.jpg) | ![demo 4](docs/images/demo-scenario4-fallback.jpg) |

> **안전 고지**: 이 시스템은 현장 안전관리자를 돕는 **보조 경보 계층**입니다. 법정 안전장치나 독립 안전 PLC(비상정지 회로)를 대체하지 않으며, 장비 인터록은 이 저장소에서 로그·웹훅까지만 구현되어 있습니다.

---

## 1. 현재 코드 구성

### 1-1. 모델과 역할

| 구분 | 모델 / 버전 | 역할 |
|---|---|---|
| **Track A** (기본) | `yolo11m-pose.pt` (Ultralytics YOLO11 medium pose) + ByteTrack | 작업자 탐지·추적(`track_id`)과 17개 관절 키포인트. 상체 기울기, 보행 속도, 시선 방향, 정지 시간 계산의 기반 |
| **Track B** (선택, 정밀 모드) | `yolov8s-worldv2.pt` (YOLO-World v2 small, 제로샷) | 별도 학습 없이 프롬프트로 건설 현장 객체 탐지: `hard hat`, `safety vest`, `safety harness lanyard`, `excavator`, `forklift`, `crane hook`, `ladder`, `scaffolding`, `open manhole`. 처음 실행 시 CLIP 텍스트 인코더(약 340MB)를 내려받음 |
| **Track C** (선택) | `models/custom_ppe.pt` (사용자 제공 PPE 파인튜닝 모델, 저장소에 미포함) | 소형 안전장구(`Hardhat`, `No-Hardhat`, `Safety-Hook`, `machinery` 등) 인식률 보강. 파일이 없으면 자동으로 건너뜀. 점검: `python scripts/check_ppe_model.py models/custom_ppe.pt` |
| **의사결정** | **Jev `jev-1.14.0`** (TypeSafe AI System One, `POST /v1/systemone`) | YOLO가 요약한 **텍스트 시퀀스**를 받아 ① 사고 전조 확률(`noul`) ② 위험 점수 0~5(`score`) ③ 권고 조치(`choice`)를 반환. 위험 정황인데 안전장구 인식이 불확실할 때만 ROI 이미지를 함께 보내 ④ 위험 영역 박스(`box`)를 받음 |

- **Jev는 영상을 직접 보는 비전 모델이 아닙니다.** 평소에는 텍스트 상태(약 1.4~2.3k 토큰)만 보내고, 이미지는 폴백일 때만 사용합니다(약 4.2k 토큰). 모델은 `jev-latest`(현재 1.13)가 아니라 `jev-1.14.0`으로 **고정**합니다. Box 질문이 1.14에서만 되기 때문입니다.
- 이미지·Box 입력은 TypeSafe 문서에서 아직 **"Dreamer-ready"(실험 단계)** 로 표기되어 있습니다.
- 테스트 환경: Python 3.14.8, ultralytics 8.4.178, torch 2.14.1, opencv-python 5.0, pydantic 2.14, httpx 0.28, fastapi 0.143, uvicorn 0.54 (macOS, Apple Silicon).

### 1-2. 처리 흐름

```
FrameSource ─► HybridVisionExtractor ─► SpatialFeatureProcessor ─► SequenceBuffer ─► JevSafetyPredictor ─► SafetyActionRouter
 최신 프레임      A 포즈+추적 (매 프레임)      PPE↔사람 매칭, 자세·시선·          5초 슬라이딩        텍스트 모드 (기본)         즉시 조치 / 관리자 확인 /
 만 유지          B/C 안전장구 (3프레임마다)    보행, 거리·TTC, 불확실성 태깅      윈도우 → 서술 JSON   이미지 폴백 (저신뢰)        정상 모니터링
```

- 영상 분석은 카메라 프레임레이트(약 25~30fps)로 돌고, Jev 호출은 **기본 1초 간격**의 비동기 호출입니다. 위험 이벤트가 감지되면 0.3초 간격까지 앞당기며, 한 번에 한 호출만 진행하므로 느린 API 응답이 영상 처리를 막지 않습니다.
- Jev는 `[T-4s: 굴착기 거리 5.8m] -> [T-2s: ... 작업자 시선이 장비 반대 방향] -> [T-0s: 예상 충돌 1.4초 전]` 같은 **시간 순서의 서술**을 보고 판단합니다. 같은 "안전모 없음"도 휴게구역에서는 낮은 확률(0.03), 장비가 접근 중이면 높은 확률(0.94)이 됩니다.

### 1-3. 디렉터리

```
main.py                  # 진입점: --web / --mock / --source
siteeye/
  config.py              # Settings, 임계값, .env 로더
  schemas.py             # Pydantic v2: 프레임 특징, 시퀀스 페이로드, Jev 요청/응답
  vision.py              # VisionModelRegistry, HybridVisionExtractor (트랙 A/B/C)
  spatial.py             # SpatialFeatureProcessor, assess() 위험 판정
  sequence.py            # SequenceBuffer (5초 슬라이딩 윈도우, 인과 서술 생성)
  jev_client.py          # JevSafetyPredictor, ROI 이미지 준비, 질문 정의
  router.py              # SafetyActionRouter, ConsoleSink / WebhookSink
  pipeline.py            # 비동기 오케스트레이션과 스로틀링
  mock.py                # 시나리오 4종, 오프라인 대체 응답기
  source.py              # RTSP/HTTP/파일/웹캠 최신 프레임 리더
  web/                   # 웹앱: app.py(API·웹소켓·보안) sessions.py(방문자 세션·한도) runner.py(분석 루프) demo.py(데모) static/(화면)
demo/street_bus.mp4      # 실제 YOLO 데모용 샘플 영상
docs/examples/           # --mock 이 만든 Jev 요청 JSON (텍스트 모드 vs 이미지 폴백)
docs/DEPLOY.md           # 공개 데모 배포 가이드 (Mac mini + Cloudflare Tunnel)
LICENSE                  # GNU AGPL-3.0
legacy/                  # 초기 버전(v1) 코드 보관 (사용하지 않음)
```

---

## 2. 위험도 판단 기준값

판단은 **2단계**입니다. ① 프레임마다 도는 가벼운 규칙(`assess()`)이 "위험 정황이 있는가, 긴급한가, 비전 근거가 약한가"를 걸러내고, ② Jev가 시간 흐름을 보고 사고 전조 확률을 판단합니다. 값은 모두 `siteeye/config.py` 의 `Settings` 에서 바꿀 수 있습니다.

### 2-1. 위험 정황 감지 (규칙, `assess()`)

| 조건 | 기준값 | 효과 |
|---|---|---|
| 장비 충돌 예상 시간(TTC) | **≤ 4.0초** (`ttc_warn_s`) | 위험 정황 |
| TTC가 긴급 | **≤ 3.0초** (`ttc_urgent_s`) | **긴급**: Jev 호출 주기 1초 → 0.3초로 단축 |
| 장비와 거리 | **≤ 4.0m** 이면서 접근 속도 > 0.2m/s (`near_distance_m`) | 위험 정황 |
| 위험구역 | 작업자 발 위치가 위험구역 다각형 안 | 위험 정황 |
| 고소 작업 | 사다리 위 **≥ 2.0m** 이면서 안전고리가 `hooked`가 아님 | 위험 정황 |
| 상체 기울기 | 사다리 위 **≥ 20°** 또는 전체 **≥ 60°** | 위험 정황 + **긴급** |

- **TTC 계산**: 거리 ÷ 접근 속도. 접근 속도가 0.3m/s 이하이면 TTC를 계산하지 않습니다.
- **거리·속도는 단안 카메라 추정**입니다(작업자 키 1.7m 기준, 오차 약 ±30%).
- **깊이 점검**: 굴착기처럼 실제 높이를 아는 장비가 사람보다 화면에서 훨씬 작게 보이면(기대 크기의 1/1.5 미만, `depth_ratio_max`) 더 멀리 있다고 보고 **TTC와 "접근 중" 판정에서 제외**하며, 서술에는 "거리 불확실"로 적습니다. 데모 영상에서 같은 깊이는 크기 비율 2.0, 훨씬 멀리 있을 때는 0.92로 갈렸습니다.
- **사다리 높이**: 사다리 바닥이 화면 밖으로 잘려 있으면 미터 값을 만들지 않고 "높이 추정 불가"로 표시합니다.

### 2-2. 인식 신뢰도 구간 (불확실성 태깅)

| 신뢰도 | 처리 |
|---|---|
| `< 0.25` | 버림 |
| `0.25 ≤ conf < 0.55` | **버리지 않고** `"uncertain (conf: 0.38)"` 로 기록해 Jev가 시간 맥락으로 보정 |
| `≥ 0.55` | 확정 (`worn`, `absent`, `hooked` 등) |
| **이미지 폴백 트리거** | 아래 "이미지 폴백 기준" 참고 |

#### 안전장구 상태의 뜻

| 상태 | 의미 |
|---|---|
| 착용 / 결속 | 찾았고 확실함 (0.55 이상) |
| 불확실 | 찾았지만 흐릿함 (0.25~0.55) |
| 손에 파지 | 머리 밖, 손목 근처에서 찾음 |
| 미착용 | "안전모 없음" 클래스를 **명시적으로 탐지**함 (Track C 모델이 있을 때만 가능) |
| **못 찾음** (`not_detected`) | 모델은 돌았지만 못 찾음. **안 썼다는 뜻이 아님** (뒷모습, 역광, 가림, 작은 크기) |
| **측정 안 함** (`not_measured`) | PPE 모델이 꺼져 있음 (빠름 모드) |

못 찾음만으로 경보를 내지 않습니다. 대신 아래 폴백으로 Jev가 확인합니다.

#### 이미지 폴백 기준

**위험 정황이 있고**(2-1) 안전장구를 YOLO가 보증하지 못할 때 작업자 크롭 이미지를 Jev에 함께 보냅니다.

| 안전장구 조건 (하나라도) | 비고 |
|---|---|
| 안전모·조끼가 **흐릿하게**(0.25~0.45) 잡힘 | |
| 안전모를 **못 찾음** 또는 **측정 안 함** | |
| 사다리 위인데 안전고리가 **결속으로 확인되지 않음** | YOLO가 못 잡는 안전고리를 Jev가 보게 함 |

폴백 때 Jev에게 추가로 묻는 것(영어 질문, 아래 측정 근거): 안전모 착용, 조끼 착용, (사다리일 때) 안전고리가 어디에도 연결되지 않고 매달려 있음. 결과는 화면의 "Jev 장구 확인"에 표시됩니다. **안전고리 미연결 ≥ 70%** 또는 **안전모 착용 ≤ 30%** 이면 확률이 낮아도 **관리자 확인(review)** 까지 올리며(자동 인터록은 아님), `ppe_flag_p` 로 조정합니다.

**측정 근거** (데모 영상 4개, 질문당 16프레임, 실제 Jev 1.14): 안전모 48/48, 조끼 24/24 정답. 안전고리 질문은 영어 32/32, 한국어 27/32라 PPE 질문은 영어로 보냅니다. 작업자 크롭은 전체 프레임과 정확도가 같으면서 토큰은 약 1/3입니다(약 1.0k vs 2.8k). 표본이 작고 AI 영상 기준이며 이미지 입력은 문서상 실험 단계입니다.

### 2-3. 안전장구 ↔ 작업자 매칭

- 안전모는 작업자 박스 **상단 30%(머리 영역)**, 조끼는 **25~75%(몸통)** 와 겹침/포함 관계로 매칭합니다. 다른 사람의 안전모는 매칭되지 않습니다.
- 안전모가 머리 영역 밖이면서 손목 키포인트 근처(박스 높이의 25% 이내)에 있으면 `held_in_hand`(손에 파지)로 봅니다.
- 안전고리는 `safety hook`이 사다리/비계와 겹치고 신뢰도 ≥ 0.55일 때만 `hooked`, 고리줄만 보이면 `unhooked`로 추정합니다.

### 2-3-1. 시퀀스에 사건으로 기록되는 기준

| 사건 | 기준 |
|---|---|
| 이동 정지 | 0.35m 이상 움직이지 않은 채 **≥ 1.5초** |
| 상체 기울기 | **≥ 15°** |
| 장비 접근 | 거리 **≤ 8m**, TTC **≤ 5초** 이면 예상 충돌 시간까지 서술 |
| 시선 방향 | 장비와 같은 방향인지 2D 시선으로 추정 (확신 못 하면 `null`) |

### 2-4. Jev 판단 → 조치 (`SafetyActionRouter`)

| 조건 | 단계 | 조치 |
|---|---|---|
| **P(전조) ≥ 0.80** 또는 **위험 점수 ≥ 4.0** | 즉시 조치 | 장비 비상정지 인터록(장면에 장비가 있을 때만) + 현장 스피커 경고방송 + 작업자 스마트워치 진동 |
| **0.50 ≤ P < 0.80** | 관리자 확인 | 안전관리자 대시보드에 스냅샷과 시퀀스 로그 전송 (사람이 확인) |
| **P < 0.50** | 정상 | 모니터링 유지 (오탐 알람 차단) |

- 같은 알림이 연속으로 나가지 않도록 스피커 10초, 스마트워치 5초, 대시보드 20초 간격으로 제한합니다. 장비 인터록은 멱등이라 제한하지 않습니다.
- 웹앱 상황판은 같은 기준으로 **정상(초록) / 주의(노랑) / 위험(빨강)** 을 표시합니다.
- Jev가 반환하는 위험 점수는 6단계(0 위험 없음 ~ 5 긴급)이고, 권고 조치는 `normal_monitoring`, `worker_smartwatch_vibration`, `site_speaker_announcement`, `equipment_emergency_stop_interlock` 중 하나입니다. 라우터는 위 확률·점수 기준을 우선하고 Jev의 권고는 화면에 함께 표시합니다.

> 이 기준값은 **합리적인 초기값**일 뿐 현장 데이터로 검증된 값이 아닙니다. 실제 현장에서는 오탐/미탐 기록을 보고 `p_immediate`, `p_review`, `ttc_warn_s`, `fallback_conf` 를 조정하세요.

---

## 3. 설치 방법

**요구사항**: Python 3.10 이상 (3.14에서 테스트), **git**(CLIP 설치에 필요), 인터넷 연결(첫 실행 시 모델 다운로드), macOS/Linux/Windows. GPU는 선택이며 Apple Silicon(MPS)과 NVIDIA(CUDA)를 자동 사용하고 없으면 CPU로 동작합니다. TypeSafe API 키가 있어야 실제 Jev를 호출할 수 있고, 없어도 오프라인 대체 응답기로 전체 흐름을 볼 수 있습니다.

```bash
git clone https://github.com/seominpapa/sitesee.git
cd sitesee

python3 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt      # 새 환경 기준 약 20초~수 분 (torch 포함)

cp .env.example .env                 # 그리고 .env 에 TYPESAFE_API_KEY=... 입력 (없어도 실행은 됨)
python main.py --web                 # http://127.0.0.1:8000
```

**처음 실행할 때 일어나는 일** (새 클론에서 검증):

| 상황 | 동작 |
|---|---|
| API 키 없음 | 오류 없이 실행. 화면 상단에 `Jev 오프라인 대체 응답기 (실제 모델 아님)`. `--source` CLI는 키 설정 또는 `--offline` 안내 후 종료 |
| 키가 틀림 (401/403) | 영상 분석은 계속되고, 화면에 "Jev API 키가 거부되었습니다" 안내가 뜨며 같은 호출을 반복하지 않음 |
| 시뮬레이션 데모 | 모델 다운로드 없이 바로 동작 |
| 첫 "영상 실처리"/웹캠 (빠름) | 포즈 모델 `yolo11m-pose.pt` (약 42MB)를 `models/`에 자동 다운로드. 화면에 "비전 모델을 불러오는 중…" 표시, 약 15초 |
| 첫 "정밀" | 제로샷 모델 `yolov8s-worldv2.pt` (약 26MB)와 CLIP 가중치(약 340MB)를 자동 다운로드, 약 20초~수 분 |
| 정밀 모델을 못 불러옴 | 포즈 추적만으로 계속 실행하고, 화면에 어떤 모델이 빠졌는지 안내 |
| Python 3.9 이하 | `Python 3.10 or newer is required` 안내 후 종료 |

- **API 키**는 `.env`(git에서 제외됨) 또는 터미널의 `export TYPESAFE_API_KEY=...` 로만 전달합니다. 코드와 로그에는 저장·출력하지 않습니다. 터미널 환경변수가 `.env`보다 우선합니다. 키는 [TypeSafe 콘솔](https://console.typesafe.ai/keys)에서 발급받습니다.
- **모델 가중치**는 저장소에 포함되지 않으며 `models/` 에 자동으로 내려받습니다(`.gitignore`).
- **환경변수**: `TYPESAFE_API_KEY`, `TYPESAFE_API_URL`(기본 `https://api.typesafe.ai/v1/systemone`), `SITEEYE_TRACKS`(기본 `A,B,C`), `SITEEYE_DEVICE`(`mps`/`cuda:0`/`cpu`), `SITEEYE_POSE_WEIGHTS`, `SITEEYE_WORLD_WEIGHTS`, `SITEEYE_PPE_WEIGHTS`, `SITEEYE_CAMERA`. 공개 모드: `SITEEYE_PUBLIC`, `SITEEYE_PUBLIC_ORIGIN`, `SITEEYE_SOURCE_URL`, `SITEEYE_MAX_SESSIONS`, `SITEEYE_MAX_DEMO_SESSIONS`, `SITEEYE_SESSION_SECONDS`, `SITEEYE_SESSION_JEV_CALLS`, `SITEEYE_DAILY_JEV_CALLS`, `SITEEYE_IP_SESSIONS_PER_HOUR`.

### 실행

```bash
python main.py --web                  # 웹앱: http://127.0.0.1:8000
python main.py --mock                 # 시나리오 4종 (카메라·GPU 불필요), 요청 JSON을 out/mock_payloads/ 에 저장
python main.py --source site.mp4      # 웹 없이 터미널에서 영상/RTSP/웹캠 번호 분석
```

---

## 4. 데모, 웹캠, CCTV 연결 방법

`python main.py --web` 실행 후 브라우저에서 **http://127.0.0.1:8000** 을 엽니다(8000번 포트가 사용 중이면 `--port 8710`). 상단 배지가 `Jev LIVE`이면 실제 모델, `Jev 오프라인 대체 응답기`이면 키가 없어 로컬 휴리스틱으로 동작 중입니다(이때 확률은 Jev의 판단이 아닙니다).

### 데모 (카메라 없이)

1. 왼쪽 **데모** 탭에서 시나리오를 고르고 **분석 시작**을 누릅니다.
2. 5종: ① 휴게구역 안전모 탈착(정상 판단) ② 굴착기 접근 + 작업자 등 돌림(즉시 조치) ③ 고소 작업 안전고리 미결속(추락 전조) ④ 역광으로 안전모 인식 저하(이미지 폴백) ⑤ 실제 영상(YOLO 실처리).
   - **①~④는 AI 생성 영상이 포함**되어 있어(`demo/s1.mp4`~`s4.mp4`) 기본이 **"영상 실처리"** 입니다. YOLO(A+B)가 영상을 실제로 탐지하고, 영상 위에 "AI 생성 영상" 배지가 표시됩니다. 처리 방식을 **"시뮬레이션"** 으로 바꾸면 장면을 그려 넣고 탐지값을 시나리오로 정해 두는 방식으로 볼 수 있습니다(YOLO 미실행, 시뮬레이션 배지). 어느 쪽이든 Jev 호출·시퀀스·조치 분기는 실제 코드가 돕니다.
   - 영상 제작 프롬프트는 [docs/DEMO_VIDEO_PROMPTS.md](docs/DEMO_VIDEO_PROMPTS.md), 파일·구역 규칙은 [demo/README.md](demo/README.md).

### 비전 정밀도 (Track A / B / C)

왼쪽 패널의 **비전 정밀도**가 어떤 모델을 돌릴지 정합니다. 웹캠·CCTV·영상 실처리 모두 적용됩니다.

| 선택 | 트랙 | 설명 |
|---|---|---|
| 빠름 | A | 사람 탐지·추적·자세 (안전모·장비는 탐지하지 않음) |
| 정밀 | A+B | + 제로샷으로 안전모·조끼·안전고리·굴착기 등 (공개 모드 기본값) |
| 최대 | A+B+C | + 현장 PPE 모델. `models/custom_ppe.pt` 가 있을 때만 선택 가능 |

- **Track C 준비**: 건설 안전용으로 학습된 YOLO 가중치(Roboflow Universe, Hugging Face 등)를 `models/custom_ppe.pt` 로 두고 `python scripts/check_ppe_model.py models/custom_ppe.pt` 로 클래스가 어떻게 매핑되는지 확인합니다. 헬멧(`Hardhat`/`No-Hardhat`), 조끼, `Safety-Hook`, `machinery` 클래스가 쓰입니다. **`.pt` 파일은 로드할 때 코드가 실행될 수 있으니 신뢰할 수 있는 출처의 파일만** 쓰고, 해당 모델의 라이선스도 확인하세요. 가중치는 저장소에 올리지 않습니다(`.gitignore`).
- 같은 PC에서 A/B/C 조합을 터미널로 지정하려면 `--tracks A,B,C`.

### CCTV (RTSP / HTTP)

**CCTV** 탭에 카메라 주소를 입력하고 분석 시작을 누릅니다. 터미널 CLI는 `python main.py --source "rtsp://..." --show`.

| 제조사/방식 | 주소 예시 |
|---|---|
| 일반 RTSP | `rtsp://아이디:비밀번호@192.168.0.10:554/stream1` |
| Hikvision | `rtsp://id:pw@192.168.0.10:554/Streaming/Channels/101` (서브스트림 `102`) |
| Dahua | `rtsp://id:pw@192.168.0.10:554/cam/realmonitor?channel=1&subtype=1` |
| HTTP 스트림 | `http://192.168.0.10:8080/video` (MJPEG/MPEG-TS 등 OpenCV·FFmpeg가 읽는 주소) |

- 먼저 `ffplay "rtsp://..."` 나 VLC로 주소가 열리는지 확인하세요. 제조사별 경로는 장비 매뉴얼을 따릅니다.
- **서브스트림(낮은 해상도)** 을 쓰면 분석이 훨씬 가볍습니다. 비밀번호에 `@ : / #` 같은 특수문자가 있으면 URL 인코딩(`%40` 등)이 필요합니다.
- RTSP는 TCP로 접속하고 끊기면 2초 간격으로 자동 재연결합니다. 신호가 없으면 화면에 "영상 신호를 기다리는 중"이 표시됩니다.
- 서버가 접속하는 것이므로 **서버 PC가 카메라와 같은 네트워크**에 있어야 합니다.

### 감시 구역 그리기

왼쪽 **감시 구역**에서 `위험구역`/`휴게구역` 버튼을 누르고 영상 위 꼭짓점을 클릭합니다(더블클릭 또는 Enter로 완료, Esc로 취소). 분석 중에도 바로 적용되며, 위험구역에 들어간 작업자는 노란색 경고로 표시됩니다. CLI는 `--danger-zone "x,y;x,y;..."`(0~1 정규화 좌표), `--rest-zone` 으로 지정합니다.

### 예시 요청 JSON

`docs/examples/` 에서 같은 시나리오의 **텍스트 모드**와 **이미지 폴백 모드** 요청을 비교할 수 있습니다(`scenario4_text.json` vs `scenario4_image.json`: 이미지 모드는 `TypeSafe-Danger-Zone-Images: true` 헤더, `assets`, `hazard_roi_box` 질문이 추가됨).

---

## 5. 그 외 필요한 사항

### 공개 데모로 호스팅하기 (Cloudflare Tunnel)

방문자가 데모를 보고 **자기 웹캠**으로 직접 시험해 보는 공개 서비스로 올리려면 `--public` 모드를 씁니다. 자세한 절차(터널 설정, launchd, 체크리스트)는 **[docs/DEPLOY.md](docs/DEPLOY.md)** 를 보세요.

```bash
python main.py --web --public        # 127.0.0.1 에서만 열리고, Cloudflare Tunnel이 HTTPS로 공개
```

| 공개 모드에서 | 동작 |
|---|---|
| 방문자 분리 | 방문자(브라우저 탭)마다 독립된 분석 세션. 동시 영상·웹캠 2명, 데모 8명 |
| 웹캠 | **방문자 자신의 웹캠**을 브라우저가 권한 요청 후 서버로 업로드해 분석 (HTTPS 필요) |
| 데모 | 각 데모는 방문자마다 **처음 한 번은 사용량 한도와 관계없이 실행 가능**. 같은 데모 반복은 시간당 한도에 포함 |
| 제한 | 세션 3분, 세션당 Jev 200회, 하루 4000회(초과 시 오프라인 대체 응답기로 표시), IP당 시간당 6회 |
| 비전 | 정밀도 선택 지원, 기본 **정밀(A+B)**. 방문자마다 별도 포즈 모델(추적 상태 분리) + **B/C는 전 세션이 공유**. 현장 PPE 모델(`custom_ppe.pt`)이 있으면 "최대"(A+B+C) 선택 가능 |
| 저장 | **영상·이미지를 디스크에 쓰지 않음.** 스냅샷도 방문자 브라우저 메모리로만 전달 |
| 차단 | CCTV 주소 입력, 서버 카메라(서버가 임의 주소에 접속하는 것을 막기 위함). CCTV는 소스를 받아 로컬에서 실행 |
| 화면 | 시작 전 안내·동의, 남은 시간, 소스 코드(AGPL-3.0) 링크 |
| 보안 | WebSocket Origin 검사, CSP 등 보안 헤더, Cloudflare 클라이언트 IP 사용 |

### 보안·개인정보

- 웹 서버는 기본 `127.0.0.1` 에만 열립니다. **인증이 없으므로** `--host 0.0.0.0` 은 신뢰할 수 있는 내부망에서만, 가능하면 리버스 프록시 인증과 함께 쓰세요.
- Jev로는 **객체 수·체류 시간·사건 같은 텍스트**가 전송됩니다. 영상 전체는 전송되지 않고, **이미지 폴백일 때만 해당 작업자 주변 ROI 크롭 1장**이 외부(TypeSafe API)로 나갑니다. 현장 영상의 개인정보(얼굴 등) 처리는 관련 법령(개인정보 보호법 등)과 사내 정책에 따라 별도 검토가 필요합니다.
- `.env` 와 키는 `.gitignore` 로 제외됩니다. 키가 노출됐다면 콘솔에서 즉시 폐기·재발급하세요.

### 비용·성능

- Jev 단가는 문서 기준 100만 토큰당 $0.042입니다. 카메라 1대를 1초 간격으로 한 시간 호출하면 약 720만 토큰(약 $0.30)입니다. 위험이 지속되는 동안은 호출이 초당 2~3회로 늘어납니다.
- Jev 응답은 약 100~550ms였습니다. Track A만 쓰면 이 Mac(Apple Silicon)에서 25fps 스트림을 실시간 처리했고(비전 약 30ms/프레임), A+B(정밀) 모드는 프레임 일부를 건너뛸 수 있습니다. 느리면 서브스트림, `--tracks A`, `aux_every` 증가를 쓰세요.

### 알려진 한계

- 거리·속도·TTC는 단안 추정이라 정밀도가 낮습니다. 정확한 값이 필요하면 카메라 캘리브레이션이 필요합니다.
- 안전고리 결속, 시선 방향은 2D 휴리스틱이며 오판할 수 있습니다. Track C(현장 PPE 파인튜닝 모델)로 보강하는 것을 권장합니다.
- Jev가 한국어 서술·질문을 얼마나 잘 처리하는지 체계적으로 검증하지 않았습니다. 품질이 낮으면 `sequence.py`, `jev_client.py` 의 문구를 영어로 바꿔 비교하세요.
- 합성 데모는 모델 성능을 보여주는 것이 아니라 **파이프라인 동작과 판단 흐름**을 보여주는 용도입니다. 실제 카메라·RTSP 환경에서의 정확도는 현장 데이터로 검증해야 합니다.
- 자동화된 테스트(pytest)는 아직 없습니다. 동작 확인은 `python main.py --mock` 과 웹 데모로 합니다.

### 문제 해결

| 증상 | 확인 |
|---|---|
| 웹 주소가 안 열림 | 서버가 실행 중인지, `http://` 로 시작하는지(브라우저가 https로 바꾸지 않았는지), 포트 충돌(`--port`) |
| `Jev 오프라인 대체 응답기` 배지 | `.env` 의 `TYPESAFE_API_KEY` 확인. 터미널 환경변수가 우선 적용됨 |
| `Jev HTTP 400` | 메시지에 원인이 표시됨. 이미지/Box는 실험 단계라 형식이 바뀔 수 있음 |
| `Jev HTTP 401` | 키가 잘못되었거나 폐기됨 |
| 서버 웹캠이 안 열림 | macOS 카메라 권한 또는 브라우저 웹캠 사용 |
| RTSP "신호를 기다리는 중" | 주소·계정·네트워크, `ffplay` 로 먼저 확인 |
| 정밀 모드 첫 시작이 느림 | CLIP 등 가중치 다운로드 중 (최초 1회) |

### 라이선스 주의

- 이 프로젝트는 **GNU AGPL-3.0**(`LICENSE`)으로 배포합니다. 서비스에 쓰는 Ultralytics YOLO가 AGPL-3.0이라, 방문자에게 웹으로 제공할 때는 서비스 전체의 소스를 공개해야 하기 때문입니다. 서비스 화면에 소스 코드 링크를 두고, 배포 코드와 공개 코드를 같게 유지하세요. 소스를 공개하지 않고 상용으로 쓰려면 Ultralytics Enterprise License가 필요합니다.
- **Ultralytics YOLO(YOLO11, YOLO-World)는 AGPL-3.0**입니다. 이를 사용하는 서비스를 외부에 제공하거나 상용 제품에 넣을 때는 소스 공개 의무 또는 Ultralytics Enterprise 라이선스를 검토해야 합니다. `demo/street_bus.mp4` 도 Ultralytics 샘플 이미지(`bus.jpg`)로 만든 데모용 영상입니다.
- 사용한 외부 서비스: TypeSafe AI Jev API ([문서](https://docs.typesafe.ai)).

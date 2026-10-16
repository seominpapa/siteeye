# 공개 데모 배포 가이드 (Mac mini + Cloudflare Tunnel)

방문자가 웹에서 **데모 시나리오**를 보고 **자기 웹캠**으로 직접 시험해 볼 수 있는 공개 서비스로 올리는 방법입니다. 공개 모드(`--public`)는 방문자 영상을 서버에 저장하지 않고, CCTV 주소 입력은 막습니다.

```
방문자 브라우저 ──HTTPS──► Cloudflare ──터널──► cloudflared (Mac mini) ──► 127.0.0.1:8000  python main.py --web --public
      (웹캠 프레임 업로드, 분석 결과 수신)                                                  └─► TypeSafe Jev API (텍스트, 필요 시 ROI 이미지 1장)
```

- 앱은 **`127.0.0.1` 에만** 열립니다. 공유기 포트포워딩이 필요 없고 외부에서 Mac mini로 직접 들어올 수 없습니다.
- HTTPS는 Cloudflare가 붙여줍니다. 브라우저 웹캠(`getUserMedia`)은 HTTPS(또는 localhost)에서만 동작하므로 이 구성이 필요합니다.

## 0. 먼저 확인할 것 (오픈 전 체크리스트)

| 항목 | 상태 |
|---|---|
| **소스 공개** | 저장소를 public으로 바꾸고, 화면 하단 "소스 코드 (AGPL-3.0)" 링크가 그 주소를 가리키는지 확인 (`SITEEYE_SOURCE_URL`). 서비스 코드와 저장소 코드를 항상 같게 유지 |
| **라이선스** | 루트의 `LICENSE`(AGPL-3.0). Ultralytics YOLO가 AGPL-3.0이라 서비스 전체를 AGPL로 공개하는 경로입니다. 비공개로 운영하려면 Ultralytics Enterprise License가 필요합니다 |
| **TypeSafe 약관** | 내 API 키로 방문자 트래픽을 보내도 되는지 [Master Customer Agreement](https://typesafe.ai/legal/mca)에 명시돼 있지 않습니다. 오픈 전에 sales@typesafe.ai에 확인하세요 |
| **개인정보** | 서비스 화면의 안내·동의 문구와 실제 동작(무저장, 외부 API 전송)이 일치하는지 검토. 국외 이전 고지 등은 전문가 확인을 권장합니다 |
| **API 키** | `.env`(gitignore)에만 두고, TypeSafe 콘솔에서 사용량 알림/한도 설정이 가능한지 확인 |
| **오프라인 시험** | 키 없이 `--offline` 로 먼저 전체 흐름을 점검 |

## 1. 앱 실행

`.env` 에 키와 공개 설정을 넣습니다(예시는 `.env.example`).

```bash
TYPESAFE_API_KEY=...
SITEEYE_PUBLIC_ORIGIN=https://demo.example.com   # WebSocket Origin 검사에 쓰는 실제 주소
SITEEYE_SOURCE_URL=https://github.com/seominpapa/sitesee
```

```bash
cd siteview   # 클론한 폴더
.venv/bin/python main.py --web --public           # 기본: http://127.0.0.1:8000
```

공개 모드의 기본 제한값 (환경변수로 변경):

| 설정 | 기본값 | 환경변수 |
|---|---|---|
| 동시 영상·웹캠·영상 실처리 분석 | 2명 (각자 별도 포즈 모델) | `SITEEYE_MAX_SESSIONS` |
| 동시 시뮬레이션 데모(YOLO 불필요) | 8명 | `SITEEYE_MAX_DEMO_SESSIONS` |
| 세션 길이 | 180초 | `SITEEYE_SESSION_SECONDS` |
| 세션당 Jev 호출 | 200회 | `SITEEYE_SESSION_JEV_CALLS` |
| 하루 Jev 호출 (초과 시 새 세션은 "오프라인 대체 응답기"로 표시) | 4000회 | `SITEEYE_DAILY_JEV_CALLS` |
| IP당 시간당 시작 횟수 | 6회 | `SITEEYE_IP_SESSIONS_PER_HOUR` |

- 각 데모는 방문자(브라우저)마다 **처음 한 번은 시간당 한도와 슬롯 한도에 걸리지 않고** 실행됩니다. 이미 본 데모를 반복하면 한도에 포함됩니다. 단, 실제 YOLO가 도는 "⑤ 실제 영상"은 슬롯이 가득 차면 잠시 기다려야 합니다.
- 위험이 지속될 때도 Jev 호출은 최소 1초 간격입니다.
- 방문자가 탭을 닫으면 10초 뒤 슬롯을 반납합니다.
- 데모 ①~④를 "영상 실처리"로 돌리면 분석 슬롯을 씁니다. 슬롯이 가득 차면 화면이 시뮬레이션으로 바꾸도록 안내하며, 시뮬레이션은 슬롯과 무관하게 언제든 실행됩니다.
- 공개 모드에서 꺼지는 것: CCTV 주소 입력, 서버 카메라, 카메라 검색 API, 스냅샷 파일 저장(`/snapshots`). 분석 이미지는 디스크에 쓰지 않고 방문자의 브라우저 메모리로만 전달됩니다.

## 2. Cloudflare Tunnel

### 2-A. 바로 시험 (도메인 없이, 임시 주소)

```bash
brew install cloudflared
cloudflared tunnel --url http://127.0.0.1:8000
```

터미널에 `https://<무작위>.trycloudflare.com` 주소가 나옵니다. 이 주소는 실행할 때마다 바뀌는 시험용이므로, 이때는 `SITEEYE_PUBLIC_ORIGIN` 을 그 주소로 맞춰 앱을 다시 실행해야 WebSocket Origin 검사를 통과합니다(비워 두면 요청의 Host와 같은 경우만 허용).

### 2-B. 내 도메인으로 상시 운영

Cloudflare에 도메인이 등록되어 있어야 합니다.

```bash
brew install cloudflared
cloudflared tunnel login                         # 브라우저에서 도메인 선택
cloudflared tunnel create siteeye                # 터널 UUID와 자격증명 JSON 생성
```

`~/.cloudflared/config.yml`:

```yaml
tunnel: <TUNNEL-UUID>
credentials-file: /Users/<사용자>/.cloudflared/<TUNNEL-UUID>.json
ingress:
  - hostname: demo.example.com
    service: http://127.0.0.1:8000
  - service: http_status:404
```

```bash
cloudflared tunnel route dns siteeye demo.example.com
cloudflared tunnel run siteeye                   # 연결 확인 후
cloudflared service install                      # 로그인 시 자동 시작 (동작 방식은 Cloudflare 문서 확인)
```

Cloudflare는 WebSocket을 기본으로 지원합니다(대시보드 Network 설정에서 WebSockets 켜짐 확인). 앱의 `/healthz` 는 헬스체크용입니다.

## 3. 앱을 항상 켜 두기 (launchd)

`~/Library/LaunchAgents/com.sitesee.demo.plist` (경로는 본인 환경에 맞게 수정):

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>com.sitesee.demo</string>
  <key>WorkingDirectory</key><string>/Users/USER/siteview</string>
  <key>ProgramArguments</key><array>
    <string>/Users/USER/siteview/.venv/bin/python</string>
    <string>main.py</string><string>--web</string><string>--public</string>
  </array>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>StandardOutPath</key><string>/Users/USER/siteview/out/app.log</string>
  <key>StandardErrorPath</key><string>/Users/USER/siteview/out/app.log</string>
</dict></plist>
```

```bash
launchctl load ~/Library/LaunchAgents/com.sitesee.demo.plist
```

앱은 프로젝트 폴더의 `.env` 를 직접 읽으므로 plist에 키를 넣지 않아도 됩니다. Mac mini가 잠들지 않도록 시스템 설정 → 에너지 절약에서 "디스플레이가 꺼져 있을 때 자동으로 잠자기 방지"를 켜세요.

## 4. 용량 가이드 (Mac mini M4 24GB)

- **공개 모드도 Track B(제로샷)를 실행합니다** (기본 "정밀" = A+B). 방문자마다 별도 포즈 모델을 쓰고(ByteTrack 상태가 모델 안에 있기 때문), B/C 모델은 상태가 없어 전 세션이 하나를 공유합니다. 서버 시작 시 미리 로드합니다. `models/custom_ppe.pt` 를 두면 "최대"(A+B+C)도 선택할 수 있습니다.
- 개발 Mac(Apple M5)에서 방문자 2명이 정밀 모드로 동시에 웹캠을 올렸을 때 각자 초당 약 10프레임의 결과를 받았고 프레임당 비전 시간은 29~32ms였습니다(B는 3프레임마다 실행).
- 분석 1건당 YOLO 포즈 모델이 프레임마다 약 30ms(개발 Mac, Apple M5 기준)이고 추론 스레드는 하나를 공유합니다. 방문자 웹캠이 초당 약 12프레임을 올리므로 **2명 동시 접속**까지를 기본값으로 잡았고, 개발 Mac에서 2명이 동시에 웹캠 스트림을 올렸을 때 각자 초당 약 9프레임의 분석 결과를 받았습니다.
- **Mac mini M4 24GB에서는 직접 측정하지 않았습니다.** 올리기 전에 `SITEEYE_MAX_SESSIONS` 를 바꿔 가며 부하를 시험해 보세요. 3명 이상은 검증하지 않았습니다.
- 메모리: 슬롯당 포즈 모델 1개(약 40MB 가중치) + 공유 B 모델(CLIP 포함, 수백 MB)이므로 24GB에서는 여유가 큽니다. 병목은 메모리가 아니라 추론 시간입니다.

## 5. Cloudflare 쪽 보호 (권장)

- **Rate limiting 규칙**: `/api/start` 경로를 IP당 분당 10회 정도로 제한 (앱 내부 제한과 이중 방어).
- **Bot Fight Mode / Turnstile**: 자동화된 접속 억제. 공개 데모에 봇이 몰리면 Jev 호출 비용이 늘 수 있습니다.
- 지인용으로만 열 때는 **Cloudflare Access**로 이메일 인증을 거는 것이 가장 간단합니다.
- 서버를 `0.0.0.0` 으로 열지 마세요. 터미널에 경고가 표시됩니다.

## 6. 운영

- 업데이트: `git pull` → 앱 재시작(`launchctl kickstart -k gui/$(id -u)/com.sitesee.demo`). 코드를 바꿨다면 공개 저장소에도 같은 코드를 반영해야 합니다.
- 로그: `out/app.log`. 영상·이미지는 기록되지 않습니다.
- 키가 노출되면 TypeSafe 콘솔에서 폐기·재발급하고 `.env` 만 교체한 뒤 앱을 재시작합니다.

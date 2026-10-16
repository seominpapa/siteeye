# 데모 영상 ①~④ AI 생성 프롬프트 (각 10초)

웹앱의 데모 ①~④는 지금 **시뮬레이션**(그려 넣은 장면 + 시나리오로 정한 탐지값)입니다. 아래 프롬프트로 만든 10초 영상을 `demo/s1.mp4` ~ `demo/s4.mp4` 로 넣으면, 화면에서 **"영상 실처리"** 를 선택해 같은 영상을 **YOLO가 실제로 탐지**한 결과로 볼 수 있습니다. 파일이 없는 데모는 계속 시뮬레이션으로 동작합니다.

> 영상 실처리에서는 모델이 실제로 본 것만 판단에 쓰입니다. AI 영상은 안전모가 작게 나오거나 장비 모양이 다르면 탐지가 안 될 수 있어서, **의도한 결과(예: ② 즉시 조치)가 그대로 나오지 않을 수 있습니다.** 이 경우 프롬프트를 조정해 다시 생성하거나 시뮬레이션을 함께 보여 주세요.

## 공통 사양 (모든 영상에 적용)

| 항목 | 값 | 이유 |
|---|---|---|
| 길이 / 비율 | **10초**, 16:9 (1280×720 이상), 24~30fps, MP4(H.264) | 웹앱은 영상을 실시간 속도로 재생하며 반복 |
| 카메라 | **고정(삼각대/폴대), 줌·패닝·컷 없음**, 약간 높은 위치의 와이드 CCTV 시점 | 카메라가 움직이면 거리·속도 계산이 틀어짐 |
| 작업자 크기 | 화면 높이의 **35~50%**, **머리부터 발끝까지 전신이 보일 것** | 포즈 키포인트와 안전모 탐지, 발 위치로 거리 계산 |
| 등장 인물 | 주인공 **1명** (다른 사람은 없거나 아주 멀리) | 추적 ID 혼동 방지 |
| 복장 | **밝은 노란 안전모, 형광 주황 조끼**, 짙은 바지 | 탐지 대비 |
| 이동 방향 | 장비·작업자의 움직임은 **화면의 좌우 방향** 위주 (카메라 쪽 정면 접근 금지) | 거리를 영상 속 2D 거리로 추정하므로 깊이 방향 움직임은 부정확 |
| 글자 | 자막, 워터마크, 시간 표시 **없음** | 생성기가 임의로 넣는 글자 억제 |
| 사고 장면 | **충돌·추락·부상은 표현하지 않음.** 접촉 직전에 끝남 | 전조(near-miss)만 보여주기 위함, 생성기 정책에도 안전 |

**공통 네거티브 프롬프트** (지원하는 도구에서 별도 입력란에 붙여 넣기):

```
camera movement, camera shake, zoom, pan, cuts, multiple shots, close-up, cinematic angle, slow motion, text, subtitles, watermark, logo, timestamp, multiple workers, crowd, blood, injury, collision, accident, cartoon, illustration, 3d render look
```

**공통 앞부분** (각 프롬프트 맨 앞에 붙이면 일관됩니다):

```
Realistic documentary footage from a fixed construction-site surveillance camera mounted on a pole, wide-angle, slightly high angle, completely static camera with no movement, no zoom, no cuts, natural lighting, 10 seconds, 16:9.
```

---

## ① 휴게구역에서 안전모를 벗는 작업자 (정상 휴식으로 판단되어야 함)

**의도**: 휴게구역에 들어가 멈춰서 안전모를 벗고 손에 든다. 규칙만 보면 "안전모 미착용"이지만 맥락상 정상 휴식.
**탐지 포인트**: 안전모가 **벗은 뒤에도 손에 들려 화면에 선명히 보여야** `손에 파지` 로 인식됩니다.

```
[공통 앞부분] A construction site rest area: a rectangular rest zone painted green on the gravel ground next to a bench and a small shaded canopy with a water cooler, site containers in the background, clear daylight. One construction worker in a bright yellow hard hat, orange high-visibility vest and dark work trousers walks in from the left side of the frame toward the rest area (seconds 0–3). He steps into the painted green zone and stops (seconds 3–5). He takes off his yellow hard hat with his right hand (seconds 5–7), then stands still holding the hard hat in his hand at his side, wiping his forehead with the other hand, helmet clearly visible, facing the camera (seconds 7–10). The worker's whole body from head to feet is always visible and occupies about 40 percent of the frame height. No other people. The camera never moves.
```

**구역 JSON**(`demo/s1.json`): 화면에 칠해진 초록 구역을 대략 감싸는 다각형을 웹앱의 "휴게구역"으로 그린 뒤 **JSON 복사**를 눌러 저장하세요.

```json
{"danger": [], "rest": [[[0.55, 0.60], [0.92, 0.60], [0.92, 0.95], [0.55, 0.95]]]}
```

---

## ② 굴착기가 접근하는데 등 돌리고 작업 중인 작업자 (협착 전조, 즉시 조치)

**의도**: 굴착기가 후진하며 다가오는데 작업자는 반대쪽을 보고 작업. 충돌 직전에 끝남.
**탐지 포인트**: 굴착기가 **노란색·전신이 보이는 전형적인 모양**이어야 하고, **화면 오른쪽에서 왼쪽으로 가로질러** 접근해야 거리가 줄어드는 것이 보입니다. 작업자는 **옆모습으로 왼쪽을 보도록** 합니다(굴착기는 오른쪽).

```
[공통 앞부분] An open earthwork area with a shallow pipe trench. A large yellow hydraulic excavator with tracks, a boom and an amber flashing beacon is on the right side of the frame and reverses slowly from right to left along the ground toward the center of the frame, covering most of the distance over the 10 seconds, moving at walking speed and stopping just before reaching the worker. On the left-center of the frame a single construction worker in a bright yellow hard hat and orange high-visibility vest stands in side profile facing LEFT, away from the excavator, bending slightly over the trench and working with a hand tool, completely unaware of the machine behind him. His whole body from head to feet is visible, about 40 percent of the frame height. The excavator is fully visible in the frame at all times. No other people. The shot ends just before the machine reaches him, no contact. The camera never moves.
```

**구역 JSON**(`demo/s2.json`): 작업자 주변 작업 반경을 위험구역으로.

```json
{"danger": [[[0.20, 0.50], [0.75, 0.50], [0.75, 0.98], [0.20, 0.98]]], "rest": []}
```

---

## ③ 안전고리를 걸지 않고 사다리에서 몸을 기울이는 작업자 (추락 전조, 즉시 조치)

**의도**: 3m 이상 높이의 사다리 위에서 안전고리를 걸지 않은 채 전동공구를 쓰다가 옆으로 몸이 기울어짐. 추락 직전에 끝남.
**탐지 포인트**: **사다리 전체(바닥부터 꼭대기)와 작업자가 한 화면**에 들어와야 높이를 계산하고, 매달린 **안전고리 줄이 어디에도 걸려 있지 않은 모습**이 보여야 `미결속` 으로 추정됩니다. 장비는 없습니다.

```
[공통 앞부분] The side wall of a two-storey building under construction. A long aluminum extension ladder leans against the wall and the entire ladder from the ground to its top is visible in the frame. One construction worker in a yellow hard hat, orange vest and a full-body safety harness stands on the ladder about 3.5 meters above the ground with his back toward the camera, using a cordless drill on the wall. A red safety lanyard hangs loose from his harness, NOT attached to the ladder or anywhere else, swinging freely. Over the 10 seconds he reaches farther and farther to his right side, his upper body leans out sideways beyond the ladder rails by about 20 to 25 degrees, one foot shifting on the rung, the ladder trembling slightly. The shot ends before he falls, he never falls. His whole body is visible. No other people, no other machinery. The camera never moves.
```

**구역 JSON**: 필요 없음(구역 없이도 판단됨).

---

## ④ 역광으로 안전모가 안 보이는 상황에서 장비가 접근 (이미지 폴백)

**의도**: 해를 정면으로 받는 역광 때문에 작업자가 실루엣으로 보이고 안전모 인식이 불확실한 상태에서 굴착기가 접근.
**탐지 포인트**: 역광이어도 **작업자와 장비 윤곽은 구분**되어야 합니다. 안전모 인식 신뢰도가 낮게(0.25~0.55) 나오면 이미지 폴백이 자동 호출되고, 아예 탐지가 안 되면 폴백 조건이 안 걸립니다. 결과는 모델과 영상에 따라 달라집니다.

```
[공통 앞부분] Late afternoon on a construction site with the low sun directly in front of the camera, strong backlight and lens flare, so the people and machines appear as dark silhouettes against a bright hazy sky with overexposed highlights, the details of the helmet are hard to see. A single construction worker stands at the center-left of the frame, seen as a dark silhouette in side profile, a hard hat shape on his head barely visible, looking down at the ground, working with his hands. A large excavator silhouette on the right side of the frame slowly approaches him from right to left along the ground, getting steadily closer over the 10 seconds and stopping a few meters away from the worker. The worker's whole body from head to feet is visible, about 40 percent of the frame height. No other people. No collision. The camera never moves.
```

**구역 JSON**: 필요 없음.

---

## 만든 뒤 확인하는 방법

1. 파일 이름을 `demo/s1.mp4` ~ `demo/s4.mp4` 로 저장합니다(구역이 필요하면 같은 이름의 `.json`도).
2. 터미널에서 먼저 모델이 무엇을 잡는지 확인합니다.

   ```bash
   python main.py --source demo/s2.mp4 --tracks A,B --show
   ```

   작업자 박스가 계속 유지되는지, 정밀 모드에서 안전모(`착용`/`불확실`)와 굴착기 박스가 나오는지 봅니다.
3. 웹앱에서 해당 데모를 선택하면 **처리 방식: 영상 실처리**가 선택 가능해집니다. 시뮬레이션과 나란히 비교해 보세요.
4. 탐지가 약하면 이렇게 다시 생성합니다.
   - 작업자를 **더 크게** (화면 높이 50%까지), 안전모 색을 **더 선명한 노랑**으로.
   - 장비는 **한 대만**, 모양이 뚜렷한 **노란 굴착기**로.
   - 카메라가 흔들리거나 컷이 바뀌면 같은 프롬프트로 다시 생성.

## 저장소와 라이선스 주의

- AI 영상 도구마다 **상업적 이용·재배포 조건**이 다릅니다. 이 저장소는 공개(AGPL-3.0)이므로, 영상을 저장소에 올리기 전에 사용한 도구의 약관을 확인하세요. 영상을 저장소에 올리고 싶지 않다면 `.gitignore`에 `demo/s*.mp4`를 추가하고 배포 서버의 `demo/` 폴더에만 두면 됩니다.
- 생성 영상에 워터마크가 자동으로 들어가는 도구가 있습니다. 탐지에는 영향이 적지만 공개 데모에서는 표시 정책을 확인하세요.

# Gemini API 안정화 패치 기록

## 배경

PaperBanana에서 Google Gemini 계열 모델, 특히 preview/image generation 모델 호출 시 무기한 hang 또는 장시간 무응답이 발생할 수 있어 안정화 패치를 적용했다.

공개 이슈/문서 조사 결과, 다음 요인이 확인되었다.

- `google-genai` SDK에서 timeout/retry 관련 이슈와 수정이 계속 발생하고 있음.
- Gemini 2.5/3 계열에서 socket stall 또는 과도한 thinking/latency 사례가 보고됨.
- 이미지 생성에서 safety/no-image edge case가 발생하면 후처리 중 hang 사례가 보고됨.
- Google 공식 Priority inference는 Tier 2/3에서 사용 가능하며, `service_tier=priority`로 요청할 수 있음.

## 변경 사항

### 1. `google-genai` dependency 업데이트

`requirements.txt`:

```txt
google-genai[aiohttp]>=1.74.0
```

목적:

- 최신 SDK의 timeout/retry fix 반영
- async 성능 개선용 `aiohttp` extra 사용

### 2. Gemini client HTTP timeout/retry 설정

`utils/generation_utils.py`에 다음 기본값을 추가했다.

- `GEMINI_HTTP_TIMEOUT_MS=600000`
- `GEMINI_HARD_TIMEOUT_SEC=660`
- `GEMINI_MAX_CONCURRENCY=2`
- retryable status: `408, 429, 500, 502, 503, 504`

Gemini client 생성 시 `types.HttpOptions(timeout=..., retry_options=...)`를 전달한다.

### 3. Wall-clock hard timeout

`call_gemini_with_retry_async()` 내부의 실제 SDK 호출을 `asyncio.wait_for()`로 감쌌다.

목적:

- SDK/transport 레벨 timeout이 동작하지 않는 socket stall 상황에서도 pipeline이 무기한 멈추지 않도록 함.

### 4. 동시성 제한

Gemini 호출을 `asyncio.Semaphore`로 제한한다.

기본 동시성은 2이며, 환경변수로 조정 가능하다.

```bash
export GEMINI_MAX_CONCURRENCY=2
```

### 5. Priority inference 옵션

환경변수 `GEMINI_SERVICE_TIER=priority`가 설정되어 있으면 request config에 `service_tier="priority"`를 적용한다.

```bash
export GEMINI_SERVICE_TIER=priority
```

주의:

- Google Gemini API Tier 2 또는 Tier 3 필요
- 표준 inference보다 비용이 높음
- quota 초과 시 standard tier로 downgrade될 수 있음

## 환경변수

- `GEMINI_HTTP_TIMEOUT_MS`: google-genai transport timeout, 기본 `600000`
- `GEMINI_HARD_TIMEOUT_SEC`: `asyncio.wait_for` hard timeout, 기본 `660`
- `GEMINI_MAX_CONCURRENCY`: Gemini 동시 호출 수, 기본 `2`
- `GEMINI_RETRY_ATTEMPTS`: SDK retry attempts, 기본 `5`
- `GEMINI_RETRY_INITIAL_DELAY_SEC`: 기본 `2.0`
- `GEMINI_RETRY_MAX_DELAY_SEC`: 기본 `60.0`
- `GEMINI_RETRY_EXP_BASE`: 기본 `2.0`
- `GEMINI_RETRY_JITTER`: 기본 `1.0`
- `GEMINI_SERVICE_TIER`: `priority` 설정 시 Google Priority inference 요청

## 검증

추가한 테스트:

- `tests/test_generation_utils_gemini_stability.py`

검증 명령:

```bash
python3 -m pytest tests/test_generation_utils_gemini_stability.py -q
python3 -m py_compile utils/generation_utils.py agents/visualizer_agent.py agents/vanilla_agent.py utils/config.py
```

결과:

- 안정화 테스트 3개 통과
- 주요 Python 파일 syntax compile 통과

## 권장 실행 설정

초기 안정성 확인 시 다음처럼 candidate/critic round를 낮춰 실행하는 것을 권장한다.

```bash
export GEMINI_SERVICE_TIER=priority
export GEMINI_MAX_CONCURRENCY=2
export GEMINI_HTTP_TIMEOUT_MS=600000
export GEMINI_HARD_TIMEOUT_SEC=660

python skill/run.py \
  --content-file method.txt \
  --caption "Figure 1: ..." \
  --task diagram \
  --output output.png \
  --num-candidates 1 \
  --max-critic-rounds 1 \
  --retrieval-setting none
```

안정화 후 `--num-candidates 2`, `--max-critic-rounds 2` 수준으로 올려 테스트한다.

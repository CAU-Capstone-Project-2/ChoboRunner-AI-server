CI/CD Pipeline. ChoboRunner AI 서버 배포 자동화 설계

> **작성·소유: 백엔드 (재민).** docs/2-4-2·docs/docker-container와 마찬가지로 백엔드가 소유하는 인프라/배포 문서다. docs/docker-container가 **"AI 서버를 어떤 Docker 이미지로 빌드하는가"** 의 단일 정답이라면, 본 문서는 그 이미지를 **"어떻게 자동으로 빌드·푸시하고 EC2에 배포하는가"** 의 단일 정답이다. 이미지 내부 구조(멀티 스테이지·healthcheck·런타임 구성)는 docs/docker-container가 정답이며 본 문서는 그것을 바꾸지 않는다.

## 1. 이 문서를 한눈에

### 이 문서가 정의하는 것

- **GitHub Actions** 워크플로우 — main 푸시 시 이미지 빌드·푸시·배포를 자동 수행하는 흐름
- **Docker Hub** 컨테이너 레지스트리 push 흐름 — 레포 `jaemin1340/capstone2-ai`
- **AWS EC2** 배포 흐름 — SSH 접속 → 기존 컨테이너 정리 → pull → `docker run`
- 파이프라인이 요구하는 **GitHub Secrets·EC2 사전 준비** 항목
- 이미지 **태깅·트리거 정책**

### 이 문서가 정의하지 않는 것

| 알고 싶은 것 | 가야 할 문서 |
| --- | --- |
| 이미지 멀티 스테이지 빌드 구조·healthcheck·런타임 구성은? | **docs/docker-container** |
| WebSocket binary frame wire format·`stop` 신호는? | **docs/2-4-2** |
| 자세 지표 계산·품질 게이트·응답 메시지 스펙은? | **docs/2-3-4 / 2-3-5 / 2-3-7** |
| EC2 인스턴스 자체의 프로비저닝(생성·VPC·IAM)은? | 본 문서 범위 밖 — §6 사전 준비로 전제만 명시 |
| Spring relay·Android 송신 측 배포 파이프라인은? | 백엔드 레포 `ChoboRunner-Backend` |

### 핵심 결정 5가지

1. **Docker Compose 미사용 (요구사항 #2).** 컨테이너 1개(uvicorn AI 서버 1개)뿐이므로 오케스트레이션 도구 없이 `docker run` 단일 명령으로 실행한다. docs/docker-container §1 "단일 책임 컨테이너"와 정합.
2. **빌드·푸시는 GitHub Actions에서 (요구사항 #1·#3).** EC2도 개발 머신도 빌드하지 않는다. GitHub Actions 러너가 `docker build` → `docker push`까지 수행하고, EC2는 pull·run만 한다 — docs/docker-container §8 "빌드 환경과 실행 환경의 분리" 그대로다.
3. **레지스트리는 Docker Hub `jaemin1340/capstone2-ai` (요구사항 #3).** docs/docker-container §9 #8의 미해결 항목(레지스트리·조직명 미정)을 본 문서가 **확정**한다.
4. **배포는 SSH로 EC2에 명령 실행 (요구사항 #4).** GitHub Actions가 EC2에 SSH 접속해 pull·정리·run 스크립트를 실행한다.
5. **무중단 아님 — 교체 배포.** 기존 컨테이너를 내리고 새 컨테이너를 띄우는 사이 수 초의 다운타임이 있다. v1 캡스톤 범위에서 허용한다(§4-4).

---

## 2. 전체 흐름

```
 개발자                GitHub                 GitHub Actions 러너            Docker Hub            AWS EC2
   │  git push main  ──▶  │  workflow 트리거  ──▶  │                              │                     │
   │                      │                      │ 0. checkout · pytest         │  (실패 시 중단)     │
   │                      │                      │ 1. checkout                  │                     │
   │                      │                      │ 2. docker login  ───────────▶│ (인증)              │
   │                      │                      │ 3. docker build -t <img> .   │                     │
   │                      │                      │ 4. docker push <img>  ──────▶│ (이미지 저장)       │
   │                      │                      │ 5. SSH ──────────────────────┼────────────────────▶│ 6. docker pull <img>
   │                      │                      │                              │◀────────────────────┤   (무인증, public)
   │                      │                      │                              │                     │ 7. 기존 컨테이너 정리
   │                      │                      │                              │                     │ 8. docker run --rm -d -p 8000:8000
```

- **빌드 환경** = GitHub Actions 러너(`ubuntu-latest`). 일회용 VM — 잡이 끝나면 폐기된다.
- **실행 환경** = AWS EC2. Docker 데몬만 있으면 되고 소스·Python·빌드 도구는 필요 없다(docs/docker-container §2).
- 위 흐름은 **main push(= PR 머지)·수동 실행** 기준이다. **main 대상 PR**에서는 0~3단계(`test`·이미지 빌드 검증)까지만 실행되고, 4~8단계(`push`·EC2 배포)는 스킵된다(§5).

---

## 3. 빌드·푸시 단계 (GitHub Actions)

### 3-1. 이미지 빌드 (요구사항 #3)

GitHub Actions 러너에서 레포를 checkout한 뒤 빌드한다.

```bash
docker build -t jaemin1340/capstone2-ai:latest .
```

- 빌드 컨텍스트(`.`)는 레포 루트 — `Dockerfile`·`.dockerignore`·`pyproject.toml`·`src/`·`server/`·`assets/models/`가 모두 포함된다(docs/docker-container §3·§4).
- MediaPipe 모델은 git에 커밋되어 있어(docs/docker-container §4) 클린 클론인 GitHub Actions 러너에서도 별도 다운로드 없이 빌드가 자기완결적으로 동작한다.
- 멀티 스테이지(builder/runtime) 구조 자체는 `Dockerfile`이 정답이며 본 문서는 변경하지 않는다.

### 3-2. Docker Hub 인증·푸시 (요구사항 #3)

```bash
docker login -u <DOCKERHUB_USERNAME> -p <DOCKERHUB_TOKEN>   # push 전 인증
docker push jaemin1340/capstone2-ai:latest
```

- `docker push`는 레포가 public이어도 인증이 필요하다 — public은 *pull* 무인증을 뜻한다(EC2의 pull은 §4-1처럼 무인증으로 동작). 따라서 빌드 잡은 push 전에 `docker login`한다.
- 자격증명은 워크플로우 YAML·로그에 평문으로 두지 않고 **GitHub Secrets**에서 주입한다(§5). 워크플로우에서는 `docker/login-action`을 쓰면 Secrets를 안전하게 처리하고 잡 종료 시 로그아웃까지 한다.
- 비밀번호 대신 **Docker Hub Access Token**(Account Settings → Security)을 쓴다 — 토큰은 권한 범위(Read/Write)를 좁히고 폐기·재발급이 쉽다.

### 3-3. 태깅

요청은 `jaemin1340/capstone2-ai`(= 암묵적으로 `:latest`)만 사용한다. docs/docker-container §8 태깅 전략은 *"`latest`는 가변이므로 배포 핀 용도로 사용 금지"* 라고 적고 있어 **요청과 긴장이 있다.**

**결정: `latest` + `<git-short-sha>` 두 태그를 함께 push**한다(작업자 확정, 2026-05-19).

| 태그 | 용도 |
| --- | --- |
| `latest` | EC2가 pull하는 배포 대상 — 요청한 흐름 유지 |
| `<git-short-sha>` | 불변 식별자 — "지금 EC2에 뜬 게 어느 커밋인가" 추적·롤백 근거 |

`latest` 하나만 쓰면 빌드가 갱신될 때마다 같은 태그가 다른 이미지를 가리켜, 사고 시 어느 커밋이 배포됐는지 식별할 수 없다. SHA 태그를 함께 붙여도 EC2 배포 스크립트는 그대로 `:latest`를 pull하므로 요청한 흐름은 바뀌지 않는다.

---

## 4. 배포 단계 (AWS EC2)

GitHub Actions가 EC2에 SSH로 접속해 아래 스크립트를 실행한다(요구사항 #4).

### 4-1. 이미지 pull

```bash
docker pull jaemin1340/capstone2-ai:latest
```

레포가 public이라 무인증으로 동작한다. `docker run`이 자동으로 pull하기도 하지만, **명시적 pull을 먼저** 해 두면 "최신 이미지 확보 실패"와 "컨테이너 실행 실패"를 분리해 진단할 수 있다.

### 4-2. ⚠️ 기존 컨테이너 정리 — "prune"의 정확한 의미

요청에 *"사전에 실행 중인 컨테이너 제거(prune)"* 라고 적혀 있으나, `docker container prune` **단독으로는 목적을 달성하지 못한다.**

- `docker container prune`은 **정지된(stopped) 컨테이너만** 제거한다. 직전 배포로 **실행 중인** 컨테이너는 건드리지 않는다.
- 정리하지 않고 새 `docker run`을 실행하면 호스트 포트 `8000`이 이미 점유되어 **포트 충돌로 실패**한다.
- `docker run --rm`의 `--rm`은 컨테이너가 *정지될 때* 자동 삭제한다는 뜻이지, 실행 중인 컨테이너를 *지금 정지*시키지는 않는다.

**올바른 정리 방법** — 컨테이너에 고정 이름(`--name`)을 주고, 배포 시 그 이름으로 강제 제거한다.

```bash
docker rm -f choborunner-ai 2>/dev/null || true   # 실행 중이면 정지+삭제, 없으면 무시
```

`docker rm -f`는 실행 중이어도 정지 후 삭제한다. `|| true`는 "처음 배포라 컨테이너가 아직 없음"인 경우에도 스크립트가 실패하지 않게 한다.

이를 위해 §4-3의 `docker run`에 **`--name choborunner-ai`를 추가**한다 — 요청 명령에는 `--name`이 없으나, 이름이 없으면 정리 단계가 어떤 컨테이너를 지울지 특정할 수 없다.

> (선택) 디스크 위생 — EC2 기본 볼륨(8 GB)은 빌드가 누적되면 오래된 이미지로 채워질 수 있다. 배포 후 `docker image prune -f`로 dangling 이미지를 정리한다. 이것이 요청의 "prune"이 의도한 바일 수도 있어 §4-3 스크립트에 포함한다.

### 4-3. 컨테이너 실행 (요구사항 #4)

```bash
docker run --rm -d \
  --name choborunner-ai \
  -p 8000:8000 \
  jaemin1340/capstone2-ai:latest
```

| 플래그 | 의미 |
| --- | --- |
| `--rm` | 컨테이너 정지 시 자동 삭제 — 정지된 컨테이너 잔여물이 쌓이지 않는다 |
| `-d` | 백그라운드(detached) 실행 |
| `--name choborunner-ai` | 고정 이름 — §4-2 정리 단계가 이 이름을 대상으로 한다 (요청 명령에 추가) |
| `-p 8000:8000` | 호스트 8000 → 컨테이너 8000(docs/docker-container §7-1) |

> docs/docker-container §7-5는 `--restart unless-stopped`·`--cpus`·`--memory`도 예시로 든다. `--rm`과 `--restart`는 함께 쓸 수 없다(상호 배타). v1은 요청대로 `--rm`을 유지하므로 재시작 정책은 적용하지 않는다 — EC2·Docker 데몬이 재부팅되면 컨테이너는 살아나지 않는다. 운영 안정성이 필요해지면 `--rm`을 빼고 `--restart unless-stopped`로 전환한다(§8 미해결 #4).

### 4-4. 다운타임

`docker rm -f`(기존 정지) → `docker run`(신규 기동) 사이에 수 초의 다운타임이 있다. 그동안 들어온 WebSocket 연결은 실패한다. v1 캡스톤 범위에서는 무중단 배포(blue-green 등)를 구현하지 않고 이 짧은 다운타임을 허용한다.

---

## 5. GitHub Actions 워크플로우 (설계 예시)

> 아래 설계대로 `.github/workflows/deploy.yml`이 레포에 구현되어 있다 — 본 블록은 그 설계 근거다. GitHub Secrets 등록·EC2 사전 준비(§6)는 별도 작업으로 남는다.

```yaml
# .github/workflows/deploy.yml
name: Build & Deploy AI Server

on:
  push:
    branches: [main]      # main 푸시(= PR 머지) 시 빌드·푸시·배포
  pull_request:
    branches: [main]      # main 대상 PR 시 test·빌드 검증 (배포는 안 함)
  workflow_dispatch:       # 수동 실행도 허용

env:
  IMAGE: jaemin1340/capstone2-ai

jobs:
  test:                                  # 배포 전 테스트 게이트 (§8 #7)
    runs-on: ubuntu-latest
    steps:
      - name: 소스 checkout
        uses: actions/checkout@v4

      - name: Python 3.11 설정
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: MediaPipe 런타임 시스템 라이브러리 설치   # Dockerfile runtime과 동일 집합
        run: |
          sudo apt-get update
          sudo apt-get install -y --no-install-recommends \
            libgl1 libglib2.0-0 libgles2 libegl1

      - name: 의존성 설치 (dev extras 포함)
        run: pip install ".[dev]"        # pyproject [project.optional-dependencies] dev

      - name: pytest 실행
        run: pytest                      # testpaths = ["tests"] (pyproject)

  build-push:
    needs: test                          # 테스트 통과 후에만 빌드·푸시
    runs-on: ubuntu-latest
    steps:
      - name: 소스 checkout
        uses: actions/checkout@v4

      - name: Docker Hub 로그인          # push 전 인증 필수 — PR에서는 스킵
        if: github.event_name != 'pull_request'
        uses: docker/login-action@v3
        with:
          username: ${{ secrets.DOCKERHUB_USERNAME }}
          password: ${{ secrets.DOCKERHUB_TOKEN }}

      - name: 이미지 빌드 (latest + git SHA)   # PR 포함 항상 실행 — 빌드 검증
        run: docker build -t $IMAGE:latest -t $IMAGE:${{ github.sha }} .

      - name: 이미지 push                # PR에서는 스킵 — :latest 덮어쓰기 방지
        if: github.event_name != 'pull_request'
        run: docker push --all-tags $IMAGE

  deploy:
    needs: build-push                    # 빌드·푸시 성공 후에만 배포
    if: github.event_name != 'pull_request'   # PR에서는 잡 전체 스킵
    runs-on: ubuntu-latest
    steps:
      - name: EC2 SSH 배포
        uses: appleboy/ssh-action@v1
        with:
          host: ${{ secrets.EC2_HOST }}
          username: ${{ secrets.EC2_USER }}
          key: ${{ secrets.EC2_SSH_KEY }}
          script: |
            IMAGE=jaemin1340/capstone2-ai
            docker pull $IMAGE:latest
            docker rm -f choborunner-ai 2>/dev/null || true
            docker run --rm -d --name choborunner-ai -p 8000:8000 $IMAGE:latest
            docker image prune -f
```

- 잡 3개: `test`(pytest) → `build-push`(빌드·푸시) → `deploy`(EC2 SSH 실행). `needs`로 직렬화한다.
- `test` 잡은 컨테이너 밖 bare 러너에서 `pytest`를 돌리므로, MediaPipe 네이티브 라이브러리가 `dlopen`하는 OpenGL ES/EGL `.so`(`libgles2`·`libegl1` 등)를 `apt`로 직접 설치한다 — `Dockerfile` runtime 스테이지와 동일한 집합. 이 스텝이 없으면 MediaPipe 추론 통합 테스트가 `libGLESv2.so.2` 부재로 실패한다.
- `test` 실패 시 `build-push`·`deploy`는 실행되지 않는다 — **검증 안 된 코드는 이미지로도, EC2로도 가지 않는다.** CLAUDE.md §9(모든 PR에 테스트 동반) 정책과 정합.
- **이벤트별 분기** — `main` 대상 **PR**에서는 `test`와 이미지 **빌드까지만** 실행해 머지 전 결함을 잡는다. `docker login`·`docker push`·`deploy`는 `if: github.event_name != 'pull_request'` 조건으로 **스킵**된다 — 미머지 코드가 `:latest`를 덮어쓰거나 운영 EC2에 배포되지 않게 한다. `main` push(= PR 머지)·수동 실행에서는 세 잡이 모두 끝까지 돈다.
- `deploy`가 `build-push`에 의존하므로 빌드·푸시 실패 시 EC2는 건드리지 않는다.
- EC2 접속은 `appleboy/ssh-action`을 예시로 든다 — SSH 방식 채택 시. AWS SSM 방식도 가능하다(§8 미해결 #1).

---

## 6. 사전 준비

### 6-1. GitHub Secrets

레포 Settings → Secrets and variables → Actions에 등록한다. 워크플로우·로그에 평문 노출 금지.

| Secret | 내용 | 비고 |
| --- | --- | --- |
| `DOCKERHUB_USERNAME` | Docker Hub 사용자명 (`jaemin1340`) | push 인증용 |
| `DOCKERHUB_TOKEN` | Docker Hub Access Token (Read/Write) | 비밀번호 대신 토큰 권장(§3-2) |
| `EC2_HOST` | EC2 퍼블릭 IP 또는 DNS | SSH 대상 |
| `EC2_USER` | SSH 사용자명 (Amazon Linux=`ec2-user`, Ubuntu=`ubuntu`) | AMI에 따라 다름 |
| `EC2_SSH_KEY` | EC2 키페어 **개인키**(`.pem` 전문) | SSH 방식 채택 시 |

### 6-2. EC2 인스턴스 측

- **Docker 설치** — EC2에 Docker Engine이 설치·기동되어 있어야 한다. SSH 사용자가 `sudo` 없이 `docker`를 쓰도록 `docker` 그룹에 추가하거나, 배포 스크립트에서 `sudo docker`를 쓴다.
- **보안 그룹(인바운드)** — 포트 `8000`(WS·healthcheck)을 Spring 백엔드가 접근할 출처에 한해 열어 둔다. SSH(`22`)는 **전체 개방(`0.0.0.0/0`)** 한다(작업자 확정) — GitHub Actions 러너 IP가 고정이 아니어서 출처를 특정 대역으로 제한할 수 없기 때문이다. ⚠️ 그 대신 SSH는 **키페어 인증 전용**이어야 한다 — `sshd_config`에서 `PasswordAuthentication no`로 비밀번호 로그인을 끈다. 키 유출 시 영향이 크므로 `EC2_SSH_KEY`는 이 배포 전용 키페어로 발급하고, 노출 시 즉시 교체한다.
- **아키텍처 정합** — GitHub Actions `ubuntu-latest` 러너는 **amd64(x86_64)**. EC2가 Graviton(arm64, `t4g`·`c7g` 등)이면 amd64 이미지는 실행되지 않거나 에뮬레이션으로 느려진다. 인스턴스 타입을 amd64 계열로 두거나, 빌드를 `docker buildx`로 멀티아키텍처화해야 한다(§8 미해결 #6).

---

## 7. 트리거·정책 요약

| 항목 | v1 정책 | 비고 |
| --- | --- | --- |
| 트리거 | `main` 푸시 자동(전체) + `main` 대상 PR(검증만) + 수동(`workflow_dispatch`) | 작업자 확정 |
| 빌드 위치 | GitHub Actions `ubuntu-latest` 러너 (amd64) | 일회용 VM, EC2와 동일 아키텍처 |
| 레지스트리 | Docker Hub `jaemin1340/capstone2-ai` (public) | docs/docker-container §9 #8 확정 |
| 배포 대상 | EC2 단일 인스턴스(amd64), 컨테이너 1개 | Compose·오케스트레이터 없음 |
| 배포 방식 | SSH 키페어로 pull·정리·run 스크립트 실행 | `appleboy/ssh-action`, 작업자 확정 |
| 다운타임 | 교체 시 수 초 허용 | 무중단 배포 미구현(§4-4) |
| 배포 전 테스트 | `test` 잡에서 `pytest` 통과 시에만 빌드·배포 | 작업자 확정, §5 |

---

## 8. 결정·미해결 항목

| # | 항목 | 결정 / 상태 |
| --- | --- | --- |
| 1 | EC2 접근 방식 | **결정** — SSH 키페어 + `appleboy/ssh-action`. SSH 22 포트 노출 정책은 #5 참조 |
| 2 | 워크플로우 트리거 | **결정** — `main` 푸시 자동(`test`·`build-push`·`deploy` 전체) + `main` 대상 PR(`test`·빌드 검증만, `push`·`deploy` 스킵) + 수동(`workflow_dispatch`) |
| 3 | 이미지 태깅 | **결정** — `latest` + `<git-short-sha>` 동시 push (§3-3) |
| 4 | 재시작 정책 | **결정** — v1은 `--rm` 유지, `--restart` 미적용. 데몬 재부팅 시 컨테이너 비복구(§4-3) |
| 5 | SSH 포트 출처 제한 | **결정** — 22 포트 전체 개방(`0.0.0.0/0`). 러너 IP 비고정 때문. 키페어 인증 전용·비밀번호 로그인 차단 전제(§6-2) |
| 6 | 빌드/실행 아키텍처 정합 | **결정** — EC2 amd64(x86_64). 러너와 동일 아키텍처라 buildx 멀티아키 불필요 |
| 7 | 배포 전 테스트 게이트 | **결정** — `test` 잡에서 `pytest` 실행, 통과 시에만 `build-push`·`deploy` 진행(§5) |
| 8 | `docker push` 인증 | **결정** — 빌드 잡에서 GitHub Secrets의 Docker Hub 자격증명으로 `docker login` 후 push. EC2 pull은 public이라 무인증(§3-2·§4-1) |
| 9 | 컨테이너 `--name` 추가 | **결정** — 정리 단계가 대상을 특정하려면 필수. `choborunner-ai`로 고정(§4-2) |
| 10 | 무중단 배포 | **v1 범위 밖** — 교체 배포의 수 초 다운타임 허용(§4-4) |

### 후속 Phase 산출물

본 문서는 설계까지다. 아래는 구현 시 필요한 작업으로, 별도 Phase로 진행한다.

- ✅ `.github/workflows/deploy.yml` 생성 완료 — `test`·`build-push`·`deploy` 3개 잡 (§5 설계대로).
- GitHub Secrets 5종 등록 (§6-1).
- EC2 인스턴스 Docker 설치·보안 그룹 구성(22 전체 개방·8000 백엔드 출처)·SSH 비밀번호 로그인 차단 (§6-2).

---

## 9. 변경 이력

- 2026-05-19 v1: 초안. GitHub Actions(빌드·푸시) + Docker Hub(`jaemin1340/capstone2-ai`) + AWS EC2(SSH 배포) 기반 CI/CD 파이프라인 설계. Docker Compose 미사용. docs/docker-container §9 #8(레지스트리 미정) 확정. "prune" 컨테이너 정리의 정확한 방법(`docker rm -f` + `--name`)을 명시. 백엔드(재민) 작성, docs/2-4-2·docs/docker-container와 동일한 인프라/배포 문서 위치.
- 2026-05-19 v1 (확정 반영): 작업자 답변으로 §8 미해결 #1·#2·#3·#6을 **결정**으로 전환 — EC2 접근=SSH 키페어(`appleboy/ssh-action`), 트리거=`main` 푸시 자동+수동, 태깅=`latest`+`<git-short-sha>`, EC2 아키텍처=amd64. §3-3·§7 표 정합 갱신.
- 2026-05-19 v1 (확정 반영 2): 잔여 미해결 #5·#7을 **결정**으로 전환 — SSH 22 포트 전체 개방(`0.0.0.0/0`, 키페어 인증 전용·비밀번호 로그인 차단 전제), 배포 전 `pytest` 게이트 도입. §5 워크플로우에 `test` 잡 추가(`test`→`build-push`→`deploy` 3잡 직렬화), §2 흐름도·§6-2·§7 표 갱신. 미해결 항목 0건.
- 2026-05-19 v1 (확정 반영 3): §1 "요구사항 정정 1건" 별도 블록 삭제 — `docker push` 인증 필요성은 §3-2 본문에 사실 정보로 통합하고, §3-2·§5·§8 #8의 "§1 정정" 참조를 정리. §8 중복 #7 행 제거. `.github/workflows/deploy.yml` 구현 완료(§5 설계대로).
- 2026-05-19 v1 (확정 반영 4): `main` 대상 **PR 검증 트리거** 추가 — `on`에 `pull_request: branches:[main]` 추가. PR에서는 `test`와 이미지 빌드까지만 실행하고 `docker login`·`docker push`·`deploy`는 `if: github.event_name != 'pull_request'`로 스킵(미머지 코드가 `:latest`·운영 EC2에 닿지 않게). §2 흐름도·§5 워크플로우·§7·§8 #2 표 정합 갱신.
- 2026-05-19 v1 (확정 반영 5): `test` 잡에 MediaPipe 런타임 시스템 라이브러리 설치 스텝 추가 — bare 러너에는 OpenGL ES/EGL `.so`가 없어 MediaPipe 추론 통합 테스트가 `libGLESv2.so.2` 부재로 실패하던 것을 수정. `Dockerfile` runtime 스테이지와 동일한 `apt` 집합(`libgl1`·`libglib2.0-0`·`libgles2`·`libegl1`)을 설치. §5 워크플로우·설명 갱신.

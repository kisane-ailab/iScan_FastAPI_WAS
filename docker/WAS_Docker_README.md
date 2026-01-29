# Docker 환경 설정 (iScan 서버)

## 직접 환경 구축하기
### 0. Synology NAS 연동하기
1. 공유 폴더 생성 또는 선택
- DSM(웹 인터페이스)에 로그인
- [제어판] > [공유 폴더]로 이동
- 공유 폴더를 생성하거나, 기존 폴더를 선택한 후 **[편집]**을 클릭
- [NFS 권한] 탭으로 이동

2. NFS 권한 추가
- [생성] 버튼 클릭
- 호스트 이름 또는 IP 주소: NFS 접근을 허용할 클라이언트의 IP 주소 또는 호스트명 입력
- [저장] 클릭하여 권한 저장

3. 서버에서 NFS 마운트
```bash
sudo mkdir /mynas
sudo mount -t nfs -o port=2049 182.208.91.210:/volume1/kisan-a31-iscan-cloud /mynas

# 부팅시 자동 마운트를 위해 아래와 같이 편집
sudo vi /etc/fstab

# In /etc/fstab:
182.208.91.210:/volume1/kisan-a31-iscan-cloud /mynas nfs port=2049 0 0

# /etc/fstab 저정후 마운트
sudo mount -a
```

### 1. ubuntu 22.04 + CUDA 12.2.0 + cudnn 8 기반 docker 이미지 가져오기
```bash
docker run -it --gpus all \
  --name iScanInstance.cheonsang_seongsu \
  -p 50000:50000 \
  -v /home/kisan-dl-admin/mynas:/mynas \
  nvcr.io/nvidia/cuda:12.2.0-devel-ubuntu22.04 /bin/bash
```
- -i 표준 입력(stdin) 활성화
- -t TTY 모드 사용, Bash 사용을 위해 필수
- --gpus nvidia gpu 사용하도록 설정
- -e 컨테이너 내 환경변수 추가
- -p host포트:container포트 매핑
- --name 실행한 docker container 이름 지정
- -v host 경로:container 경로
- *nvcr.io/nvidia/cuda:12.2.0-devel-ubuntu22.04* : nvidia docker 이미지
- /bin/bash bash 셀에 진입

명령어 실행 직후 docker container 내부로 이동

### 2. Python 3.10 + pip 설치
```bash
apt update && apt install -y software-properties-common vim
add-apt-repository ppa:deadsnakes/ppa -y
apt update
apt install -y python3.10 python3.10-dev python3.10-distutils python3-pip git curl
ln -sf /usr/bin/python3.10 /usr/bin/python
ln -sf /usr/bin/pip3 /usr/bin/pip
python -m pip install pip --upgrade
```

### 3. PyTorch + torchvision + torchaudio 설치 (CUDA 12.1 휠 사용)
```bash
pip install torch==2.3.0+cu121 torchvision==0.18.0+cu121 torchaudio==2.3.0+cu121 \
  -f https://download.pytorch.org/whl/torch_stable.html

```
CUDA 12.2은 아직 공식적으로는 +cu121 휠을 사용함. PyTorch는 하위 호환으로 잘 동작함.

### 4. iScan_FastAPI_WAS Requirement 설치
```bash
pip install fastapi uvicorn aiohttp python-multipart requests psutil pyzmq pyzipper cryptography
```

### 5. Artis_AI Requirement 설치
```bash
# mmcv 설치 (오래걸림)
pip install mmcv-full==1.7.2

# 기타 필요 패키지 설치
apt update && apt install -y libgl1
pip install scipy==1.8.0 terminaltables==3.1.10 pycocotools==2.0.8
pip install opt-einsum==3.3.0
```

### 6. Docker container 이미지 저장 (필수X)
- docker에서 빠져나오기
```bash
exit
```

- 호스트에서 실행
```bash
# docker 시작
# docker start {docker container 이름}
docker commit iScanInstance.cheonsang_seongsu

# docker container 이미지로 commit
# docker commit {docker container 이름} {docker container를 저장할 이름}
docker commit iScanInstance.cheonsang_seongsu artis-ai-image

# docker image에서 컨테이너 만들기
docker run -it --gpus all --name artis-ai-container \
  -p 50000:50000 \
  -v /home/kisan-dl-admin/mynas:/mynas \
  artis-ai-image /bin/bash
  
```

### 7. Docker로 필요한 파일 복사
```bash
# docker cp {복사 파일} {docker contrainer 이름}:{contrainer 저장 경로}
docker cp iScan_FastAPI_WAS/ iScanInstance.cheonsang_seongsu:/iScan_FastAPI_WAS
```

### 8. 서버 실행
```bash
# docker 실행
docker exec -it iScanInstance.cheonsang_seongsu /bin/bash

# workdir 이동
cd /iScan_FastAPI_WAS

# 서버 실행
nohup uvicorn app.main:app --reload --host 0.0.0.0 --port 50000 &> uvicorn.log &
```

## 컨테이너 이름 규칙
*iScanInstance.{vendor}*

*iScanInstance.{vendor}.{db_key}*

설정 예시)
- iScanInstance.cheonsang_seongsu
- iScanInstance.Kisan_Baguette.Kisan_Baguette_TOP_ES3
- iScanInstance.illiers_combray
- iScanInstance.hjhbakery

## Docker 저장 위치 변경하기

Docker의 기본 경로는 /var/lib/docker

### 1. HDD 마운트
```bash
## 폴더 생성
mkdir /local_hdd

## HDD 장치명 확인
sudo fdisk -l

## ext4로 포맷
sudo mkfs.ext4 /dev/sdb2

## HDD 마운트
sudo mount /dev/sdb2 /local_hdd
```

### 2. Docker 서비스 중지
```bash
sudo systemctl stop docker
```

### 3. 기존 Docker 데이터 이동
```bash
sudo rsync -aP /var/lib/docker/ /local_hdd/docker/
```
🔁 rsync는 퍼미션, 심볼릭 링크, 소유권까지 안전하게 복사함
⏳ 시간이 오래 걸림

### 4. Docker 설정 변경
Docker 데몬 설정 파일 열기

```bash
sudo vi /etc/docker/daemon.json
```

아래 항목 추가 또는 생성:
```json
{
  "data-root": "/local_hdd/docker",
  "storage-driver": "overlay2"
}
```

### 5. /var/lib/docker 백업 & 비우기 (선택)
```bash
# 백업
sudo mv /var/lib/docker /var/lib/docker.bak

# 삭제
sudo rm -rf /var/lib/docker
```

### 6. Docker 재시작
```bash
sudo systemctl restart docker
```

정상 동작 확인
```bash
docker info | grep "Docker Root Dir"
```

## 기타
### MPS(Multi-Process Service) 활성화 (중단)
- MPS는 CUDA 프로세스들이 GPU의 SM(SM = Streaming Multiprocessors)을 공유할 수 있도록 하여 병렬 실행 효율을 높여줌
- RTX 3090은 MPS 지원 가능(GA100 계열이 아니어도 가능), 단 MIG는 미지원
- MPS는 GPU 전체 단위로 활성화되며, 컨테이너별로 독립적으로 설정할 수 없음 → 호스트에서 관리

1) 사전조건
- NVIDIA 드라이버 설치 및 nvidia-docker2 구성 완료
- NVIDIA Container Toolkit 설치 (docker run 시 --gpus 옵션 사용 가능해야 함)
- CUDA >= 10.x 이상 (권장 CUDA 11.x 이상)
- 컨테이너들은 동일 GPU (예: GPU 0)을 공유

2) MPS 활성화
```bash
# GPU가 MPS를 지원하는지 확인
nvidia-smi -q | grep "MPS"

# 출력 예:
Multi-Process Service : Supported

# 활성화
sudo nvidia-cuda-mps-control -d
```
nvidia-smi -q | grep "MPS" 실행 결과 결과가 없을 수 있음

이는 MPS를 지원하지 않거나 현재 GPU 드라이버가 MPS 관련 정보를 노출하지 않는 경우 일 수 있음

이때는 MPS 기능을 직접 활성화 해보고 사용 여부를 확인해야 함

3) 환경변수 지정
- CUDA_MPS_PIPE_DIRECTORY
- CUDA_MPS_LOG_DIRECTORY

```bash
# Docker run 시
docker run --gpus "device=0"
-e CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
-e CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps
your_image_name

# 또는 Dockerfile 내에서
ENV CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
ENV CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps
```

4) MPS 활성화 확인
```bash
printenv | grep CUDA_MPS
```
또는 추론 실행 시 nvidia-smi에서 MPS Server 밑에 프로세스로 표시되는지 확인

5) MPS 비활성화
```bash
echo quit | sudo nvidia-cuda-mps-control
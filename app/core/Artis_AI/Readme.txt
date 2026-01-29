0. 환경 설정
	1) Cuda toolkit 설치
		https://developer.nvidia.com/cuda-11-7-0-download-archive?target_os=Windows&target_arch=x86_64
		- 설치후 cmd에서 nvidia-smi를 쳐서 cuda version 확인(11.7)

	2) Microsoft store 등을 통해 Python 3.9 설치

1. 파이썬 패키지 설치
	1) Pytorch: cmd에서 cuda 버전에 맞는 pytorch 설치하기
		pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu117

	2) MMDetection 라이브러리 설치
		pip install openmim
		python -m mim install mmcv-full

	3) 기타 라이브러리 설치
		pip install pyzmq
		pip install -r requirements.txt

2. 파이썬 예제 프로그램 실행
	1) 명령어를 주고 결과를 받을 zeroMQ 서버 실행
		python server.py port_num

	2) 추론 프로그램 실행
		python inference.py port_num config_file_path cal_file_path

3. zeroMQ 메시지 종류
	1) Kiosk -> Python
		웜업 시작 : 0x02(stx)|0x01(cmd)|10(횟수제한)|1(det_time제한)|0.1(cls_time제한)|1(dep_time제한)|0x03(etx)
		추론 시작 : 0x02(stx)|0x02(cmd)|1(영상개수)|영상절대경로+파일이름|디버그(True/False)|디버그영상저장경로|동작모드|0x03(etx)
			※ 동작 모드
			UserRun : 일반
			NewItem : 신규 아이템 추가
			Base    : 바닥면 또는 쟁반 추가
			CalCam  : 카메라 보정

		Alive 체크 : 0x02(stx)|0xEE(cmd)|0x03(etx)
		Quit 요청 : 0x02(stx)|0xFF(cmd)|0x03(etx)

		모델 업데이트 시작: 0x02(stx)|0xF5(cmd)|설정파일절대경로+파일이름|0x03(etx)

	2) Python -> Kiosk
		웜업 OK : 0x02(stx)|0x01(cmd)|OK(result)|0.36(det_time)|0.05(cls_time)|0.36(dep_time)|0.77(total_time)|0x03(etx)
		웜업 NG : 0x02(stx)|0x01(cmd)|NG(result)|1(det_time=1초이상)|0.1(cls_time=0.1초이상)|1(dep_time=1초이상)|2.1(total_time=2.1초이상)|0x03(etx)

		#추론 OK : 0x02(stx)|0x02(cmd)|OK(결과)|인식된 물체 개수(최대 50개)|인덱스|스코어|좌표 4개|...|0x03(etx)
		### 20250120 Depth Inference 통신 추가 ###
		추론 OK : 0x02(stx)|0x02(cmd)|OK(결과)|인식된 물체 개수(최대 50개)|인덱스|스코어|좌표 4개|...|Depth 기반 인식된 물체 개수(최대 50개)|인덱스|스코어|좌표 4개|...|0x03(etx)
		추론 NG : 0x02(stx)|0x02(cmd)|NG(결과)|0x01~0x0B(에러코드)|0x03(etx)

		모델 업데이트 응답 OK : 0x02(stx)|0xF5(cmd)|OK(결과)|0x03(etx)
		모델 업데이트 응답 NG : 0x02(stx)|0xF5(cmd)|NG(결과)|0x03(etx)

		2-1) 추론 에러 코드
			0x01 프로토콜 에러, 전송된 스트링이 0x02로 시작되지 않거나, 0x03으로 종료되지 않을 때
			0x02 정의되지 않은 명령 실행 (현재는 0x01 추론시작이 아니면 오류)
			0x03 추론불가능한 영상 개수 (현재는 1이 아니면 오류)
			0x04 입력 절대경로 수 < 영상 개수일 때
			0x05 입력 이미지가 존재하지 않을 때, 혹은 절대경로 오류
			0x06 디버깅 유무가 없을 때
			0x07 디버깅 옵션이 True일때 이미지 저장경로가 누락되었을 때
			0x08 입력 이미지 파일이 비어있거나 깨져있을 때(문제되는 이미지 경로와 함께 보냄)
			0x09 AI 모델이 검출한 영역 이외에 사물이 있을 때 (Depth 기반 검출)
			0x0A 사물이 서로 겹쳐있을 때 (Depth 기반 검출)    
				=> 겹침 검출시 : 0x02(stx)|0x02(cmd)|NG(결과)|0x0A(에러코드)|인덱스|좌표 4개|...|0x03(etx)
			0x0B Depth가 정상적으로 생성되지 않았을 때

		Alive 체크 OK : 0x02(stx)|0xEE(cmd)|OK(결과)|0x03(etx)
		Alive 체크 NG : 0x02(stx)|0xEE(cmd)|NG(결과)|0x01~0x(에러코드)|0x03(etx)

		Alive 체크 에러 코드
		0x01 
		...

		Quit 요청 OK : 0x02(stx)|0xFF(cmd)|OK(결과)|0x03(etx) -> 전송 이후 프로그램 종료
		Quit 요청 NG : 0x02(stx)|0xFF(cmd)|NG(결과)|0x03(etx)



### 🔗 출처 및 라이선스

이 프로젝트는 **두산로보틱스(Doosan Robotics Inc.)**에서 배포한 ROS 2 패키지를 기반으로 합니다.  
해당 소스코드는 [BSD 3-Clause License](https://opensource.org/licenses/BSD-3-Clause)에 따라 공개되어 있으며,  
본 저장소 또한 동일한 라이선스를 따릅니다. 자세한 내용은 `LICENSE` 파일을 참고하시기 바랍니다.

> ⚠️ 본 저장소는 두산로보틱스의 공식 저장소가 아니며, 비공식적으로 일부 수정 및 구성을 포함하고 있습니다.  
> 공식 자료는 [두산로보틱스 공식 홈페이지](http://www.doosanrobotics.com/kr/)를 참고해 주세요.   
> github (https://github.com/DoosanRobotics/doosan-robot2)


## 1. 프로젝트 개요

**피망 자동 수확 시스템**은 **수확 자동화** 를 통해 농촌 인력난 및 고령화 문제를 완화하고자 개발된 시스템입니다.

### 작업 환경
<img src="https://github.com/daeyeong-choi/Bellpepper-Harvest-System-with-co-robot/blob/main/image/workspace.png" width="50%" height="50%" title="px(픽셀) 크기 설정" alt="project_management"></img> 

## 2. 개발 기간 및 인원

**진행 일자: 25.5.23(금) ~ 25.6.5(목) (14일)**

|이름|담당 업무|
|--|--|
| 최대영 | 전반적인 코드 작성 및 총괄 |
| 장** | 분류 코드 작성 및 영상 제작 |
| 유** | 분류 및 재고 코드 작성 및 검토 |
| 김**| 분류 코드 및 보고서 작성 |

---

## 4. SKILLS
### **Development Environment**
<div align=left>
  
  ![Ubuntu](https://img.shields.io/badge/Ubuntu-E95420?style=for-the-badge&logo=ubuntu&logoColor=white)
  ![Visual Studio Code](https://img.shields.io/badge/Visual%20Studio%20Code-0078d7.svg?style=for-the-badge&logo=visual-studio-code&logoColor=white)
</div>

### **Robotics**
![ROS](https://img.shields.io/badge/ros-%230A0FF9.svg?style=for-the-badge&logo=ros&logoColor=white)   

### **Programming Languages**
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)   

### **AI & Computer Vision**
<div align=left>
  
  ![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
  ![OpenCV](https://img.shields.io/badge/opencv-%23white.svg?style=for-the-badge&logo=opencv&logoColor=white)
  ![YOLO](https://img.shields.io/badge/YOLO-111F68?style=for-the-badge&logo=YOLO&logoColor=white)
</div>

---
## 4. 시스템 구성 및 기능

### 하드웨어 구성
#### **Robot**
- Doosan Robotics m0609, OnRobot RG2 Gripper
<p align="left">
  <img src="https://github.com/daeyeong-choi/Bellpepper-Harvest-System-with-co-robot/blob/main/image/Doosan-Robotics-M0609-Cobot.png" width="200" />
  <img src="https://github.com/daeyeong-choi/WMS/blob/main/images/rg2.png" width="100" />
</p>

#### **Vision Camera**
- Intel RealSense D435i
<img src="https://github.com/daeyeong-choi/Bellpepper-Harvest-System-with-co-robot/blob/main/image/IntelR-RealSense" width="150" />

#### **Mic**
- Labtop

### AI 비전 시스템
- **YOLOv11n 기반 객체 탐지**  

### 로봇 제어 시스템
- **rclpy 기반 ROS2 제어 패키지 구성**  
- **Doosan ROS2 API 사용**

###  Voice AI 시스템

| 구성 | 사용 기술 |
|------|-----------|
| Wakeword 감지 | OpenWakeWord (TFLite) |
| 음성 인식 | Whisper-1 |
| 의미 분석/추천 | GPT-4o |

---

## 4. 동작 흐름

1. wake up word "hello rokey" 인식 후 동작 지시
2. 동작 지시 문장에서 object(pimang 정상 피망, rotten 불량 피망), target(pos1 정상 피망 박스, pos2 불량 피망 박스) 키워드 추출
3. 추출된 키워드에 따라 동작(1.수확해 2.불량만 수확 3.정상 n개만 수확해(개수 지정 수확은 미구현))
4. 탐지 구역으로 이동 후 수확 실행
5. 그리퍼를 통해 수확 후 분류하며 소독 모션 진행(전염 방지)
6. 탐지 실패 시 다른 탐지 구역으로 이동하며 탐지 구역을 전부 돌면 프로그램 종료
   
---

## 5. 시스템 아키텍처

### 전체 시스템 동작 흐름
<img width="850" alt="image" src="https://github.com/daeyeong-choi/Bellpepper-Harvest-System-with-co-robot/blob/main/image/flow2.png" />


### 로봇 동작 흐름

<img width="783" alt="image" src="https://github.com/daeyeong-choi/Bellpepper-Harvest-System-with-co-robot/blob/main/image/flowchart.png" />

---

### 6. 주요 코드 및 설명
project_code: dsr_rokey/pick_and_place_voice

-주요 동작 코드
![robot_control](https://github.com/daeyeong-choi/Bellpepper-Harvest-System-with-co-robot/blob/main/robot_control_final.py)

```python
##추출된 키워드 리스트에서 인덱싱하여 동작 조건 판단
if object_list[0] == 'pimang'  and len(target_list) == 2:
##탐지 구역 이동 구역은 총 4곳으로 나눠져 있으며 object탐지 실패시 구역 리스트에서 인덱싱 숫자를 +1 하여 구역 변경
    while self.section_num < 5:
        if self.state == 'failed': 
            self.section_num += 1
            self.state = 'normal'
            if self.section_num >= 4:
                self.get_logger().warn("all section detected")
                break

        mwait()
        self.get_logger().warn("change section")
        movel(self.scan_section[self.section_num], vel=VELOCITY, acc=ACC)

        target_pos = self.get_target_pos('pimang')##get_target_pos는 object 위치를 추정하여 반환하는 코드
        if target_pos is None:
            self.get_logger().warn("rotten pimang dection")
            target_pos_rotten = self.get_target_pos('rotten')
            if target_pos_rotten is None:
                self.state = 'failed'
                continue 

            else:                   
                target_pos_rotten=self.app_target(target_pos_rotten)
                self.move_target(target_pos_rotten)
                mwait()
                self.pick_to_put_place_r(self.position_map[target_list[1]])#불량 박스로 이동하는 코드
                self.sterilize() #소독 모션 코드
                continue

        else:
            target_pos=self.app_target(target_pos)
            self.get_logger().info(f"target position: {target_pos}")
            self.move_target(target_pos)
            mwait()
            self.pick_to_put_place_n(self.position_map[target_list[0]])#정상 박스로 이동하는 코드
            self.sterilize()
```
---

## 7. 기대 효과

### 기대 효과

- 농업인 체력 부담 완화
- 고용 중 발생하는 고충 완화
- YOLO모델 학습과 로직 수정을 통한 타 작물 수확 가능
  
---

## 8. 🎥 Demo video link
[![YouTube](https://img.shields.io/badge/YouTube-FF0000?logo=youtube&logoColor=white)](https://youtube.com/shorts/V1mgkM89k8Y?feature=share)

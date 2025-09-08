##   피망 자동 수확 시스템(작성 중..)
- 사용자의 음성 명령을 인식하여 수확을 실행, 피망을 감지하고, 병든 피망과 정상 피망을 구분 하여 자동 수확 수행
<img src="https://github.com/daeyeong-choi/WMS/blob/main/images/workspace.png" width="600" />


##   개발 기간 및 개발 인원
-2025.05.09~2025.05.22
|  | 역할 | 담당 업무|
| --- | --- | --- |
| 최대영 | 팀장 | 동작 코드 작성 및 총괄 |
| 장** | 팀원 | datasets 수집 labeling, 모델 학습, 영상 제작 |
| 유** | 팀원 | datasets 수집 및 labeling |
| 김**| 팀원 | 코드 및 보고서 작성, datasets labeling, 보고서 작성 |
  
## 개발환경  및 장비
![Ubuntu](https://img.shields.io/badge/Ubuntu-22.04-E95420?logo=ubuntu&logoColor=white)
![ROS2](https://img.shields.io/badge/ROS2-Humble-22314E?logo=ros&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white) 
<p align="left">
  <img src="https://github.com/daeyeong-choi/WMS/blob/main/images/Doosan-Robotics-M0609-Cobot.png" width="400" />
  <img src="https://github.com/daeyeong-choi/WMS/blob/main/images/rg2.png" width="100" />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <img src="https://github.com/daeyeong-choi/WMS/blob/main/images/%EC%8A%A4%ED%81%AC%EB%A6%B0%EC%83%B7%202025-09-04%2017-47-25.png" width="400" />
</p>


두산 로보틱스 M0609, OnRobot RG2 Gripper, Lego blocks

## 작업 환경 가정
- 수직재배에 특화된 작물인 파프리카로 선정하여 수확 및 분류 가능한 스마트팜 구현
- 음성명령 → 객체인식 → 수확 → 분류  순서로 작업 수행
- 파프리카 한 개 재배 후 소독 작업 진행
<p align="left">
  <img src="https://github.com/daeyeong-choi/WMS/blob/main/images/workspace.png" width="470" />
  <img src="https://github.com/daeyeong-choi/WMS/blob/main/images/%EC%8A%A4%ED%81%AC%EB%A6%B0%EC%83%B7%202025-09-04%2017-51-18.png" width="400" />
</p>

## 기능   

1. 5가지 종류의 레고블록 분류 후 적재
2. 원하는 블록 출고
3. 재고 확인
  ## 로직 블록도
- 전체 동작 로직 블록도
<img src="https://github.com/daeyeong-choi/WMS/blob/main/images/%EC%8A%A4%ED%81%AC%EB%A6%B0%EC%83%B7%202025-09-04%2018-55-05.png" width="500" />
- 분류 상세 로직 블록도 
<img src="https://github.com/daeyeong-choi/WMS/blob/main/images/%EC%8A%A4%ED%81%AC%EB%A6%B0%EC%83%B7%202025-09-04%2019-15-19.png" width="500" />

## 주요 코드

![코드 베이스](https://github.com/DoosanRobotics/doosan-robot2)

- 시스템 개요

1. **블록별 객체 생성**  
   생성 시 필요한 정보:
   - 블록 이름
   - 창고 이름
   - 블록을 쌓을 위치
   - 쌓기 전 이동할 위치
   - 쌓은 후 이동할 위치
   - 초기 재고 수량
   - 블록을 빼낼 위치

2. **블록 분류 및 동작 수행**  
   - 블록을 분류하거나 사용자가 입력한 블록 모양에 따라  
     해당 위치에서 블록을 쌓거나 이동하는 동작.

- 분류 및 입고 코드 및 동작 사진
<img src="https://github.com/daeyeong-choi/WMS/blob/main/images/inbound_code.png" width="600" />
<img src="https://github.com/daeyeong-choi/WMS/blob/main/images/inbound_code2.png" width="600" />

```python
def __move_to_pos_stack(self, target_pos, target_pos_up):

        z1=posx(661.400, -48.260,70.550, 126.42, 179.69, 35.9)# 높이 측정하려 내려갈 위치
        print("moving to target_pos_up")
        movel(target_pos_up, vel=VELOCITY, acc=ACC, mod=0)
        print("target_pos_up:", target_pos_up)
        mwait()

        if self.stock == 1:
            print("moving to target_pos")
            movel(target_pos[0], vel=VELOCITY, acc=ACC, mod=0)
            print("target_pos:", target_pos[0])
            mwait()

        elif self.stock ==2:
            print("moving to target_pos")
            movel(target_pos[1], vel=VELOCITY, acc=ACC, mod=0)
            print("target_pos:", target_pos[1])
            mwait()

        ##########################################################힘제어
        time.sleep(0.5)
        task_compliance_ctrl(stx=[500, 500, 500, 100, 100, 100])
        time.sleep(1)
        set_desired_force(fd=[0, 0, -15, 0, 0, 0], dir=[0, 0, 1, 0, 0, 0], mod=DR_FC_MOD_REL)
        time.sleep(1)
        force_condition = check_force_condition(DR_AXIS_Z, max=15)
        time.sleep(1)

        print("force_condition Start : ", force_condition)
        while (force_condition > -1):
            force_condition = check_force_condition(DR_AXIS_Z, max=15)
            time.sleep(2)
        time.sleep(1)
        release_force()
        release_compliance_ctrl()
        release()
        ###########################################################
        movel(target_pos_up, vel=VELOCITY, acc=ACC, mod=0)
        movel(z1, vel=VELOCITY, acc=ACC)

def stack(self):
    if not self.position:
        print("No position set for stacking.")
        return
    self.__move_to_pos_stack(self.position, self.position_up)
    self.stacked=True
    self.set_stock(self.stock + 1)
    print(f'stock: {self.stock}')
```

- 출고 코드 및 동작 사진
<img src="https://github.com/daeyeong-choi/WMS/blob/main/images/outbound_code.png" width="600" />

```python
def __move_to_pos_unstack(self, target_pos, target_pos_up, unstack_pos_list):
        p_out=posx(647.28, 198.80, 51.03, 4.75, -178.89, -84.33)
        p_out_up=posx(647.28, 198.80, 120.03, 4.75, -178.89, -84.33)
        release()
        print("moving to target_pos_up")
        movel(target_pos_up, vel=VELOCITY, acc=ACC, mod=0)
        print("target_pos_up:", target_pos_up)
        mwait()


        if self.stock == 2:
            print("moving to grip")
            movel(unstack_pos_list[1][0], vel=VELOCITY, acc=ACC, mod=0)
            grip()
            wait(1)
            movel(unstack_pos_list[1][1], vel=VELOCITY, acc=ACC, mod=0)
            movel(unstack_pos_list[1][2], vel=VELOCITY, acc=ACC, mod=0)
             
        elif self.stock ==3:
            print("moving to grip")
            movel(unstack_pos_list[0][0], vel=VELOCITY, acc=ACC, mod=0)
            grip()
            wait(1)
            movel(unstack_pos_list[0][1], vel=VELOCITY, acc=ACC, mod=0)
            movel(unstack_pos_list[0][2], vel=VELOCITY, acc=ACC, mod=0)

        movel(target_pos_up, vel=VELOCITY, acc=ACC, mod=0)
        movel(p_out_up, vel=VELOCITY, acc=ACC)
        movel(p_out, vel=VELOCITY, acc=ACC)
        release()

def unstack(self):
    if self.stock <= 0:
        print(f"Box {self.id} is out of stock!")
        return
    
    print(f'stock: {self.stock}')
    self.__move_to_pos_unstack(self.position, self.position_up, self.unstack_pos_list)
    self.set_stock = self.stock - 1
    print(f'stock: {self.stock}')
```

- 재고 체크 코드 및 사진

```python
def info(self):
    return f"id : {self.id}\nstock : {self.stock}\nposition : {self.pos_id}\n=====\n"
~~~
elif mode_select == 3:
    while True:
        movej(HOME_READY, vel=VELOCITY, acc=ACC)
        select_wh = int(input('select_wh_number :'))

        if select_wh == 1:  # small sq
            print(aa.info())
        elif select_wh == 2:  # medium sq
            print(bb.info())
        elif select_wh == 3:  # large sq
            print(cc.info())
        elif select_wh == 4:  # round sq
            print(dd.info())
        elif select_wh == 5:  # high sq
            print(ee.info())
        elif select_wh == 6:  # change to mode_select
            break
        else:
            print('wrong number!')
```
<img src="https://github.com/daeyeong-choi/WMS/blob/main/images/check.png" width="600" />

## Demo video link
![YouTube](https://img.shields.io/badge/YouTube-FF0000?logo=youtube&logoColor=white)](https://youtube.com/shorts/2GCx_hT8v0I?feature=share)

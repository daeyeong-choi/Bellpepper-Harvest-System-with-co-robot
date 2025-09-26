# ros2 service call /get_keyword std_srvs/srv/Trigger "{}"

import os
import rclpy
import pyaudio
from rclpy.node import Node

from ament_index_python.packages import get_package_share_directory
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

from std_srvs.srv import Trigger
from voice_processing.MicController import MicController, MicConfig

from voice_processing.wakeup_word import WakeupWord
from voice_processing.stt import STT

############ Package Path & Environment Setting ############
current_dir = os.getcwd()
package_path = get_package_share_directory("pick_and_place_voice")

is_laod = load_dotenv(dotenv_path=os.path.join(f"{package_path}/resource/.env"))
openai_api_key = os.getenv("OPENAI_API_KEY")


############ AI Processor ############
# class AIProcessor:
#     def __init__(self):



############ GetKeyword Node ############
class GetKeyword(Node):
    def __init__(self):


        self.llm = ChatOpenAI(
            model="gpt-4o", temperature=0.5, openai_api_key=openai_api_key
        )

        prompt_content = """
            당신은 사용자의 문장에서 특정 물체와 목적지를 추출해야 합니다.

            <목표>
            - 문장에서 다음 리스트에 포함된 물체를 최대한 정확히 추출하세요.
            - 문장에 등장하는 도구의 목적지(어디로 옮기라고 했는지)도 함께 추출하세요.

            <물체 리스트>
            - pimang, stem, rotten
            <위치 리스트>
            - 피망 바구니 = pos1, 병든 피망 바구니 = pos2
            <개수 표현>
            - 1개, 2개, 3개, 4개처럼 숫자 + "개" 형식으로 표현되며, 생략된 경우 1개로 간주합니다.
            - "하나", "두 개", "세 개", "네 개" 같은 표현도 숫자로 변환해 반영합니다.

            <출력 형식>
            - 다음 형식을 반드시 따르세요: [물체1 물체2 ... / ...개수/ pos1 pos2 ...]
            - 물체와 위치는 각각 공백으로 구분
            - 물체가 없으면 앞쪽은 공백 없이 비우고, 목적지가 없으면 '/' 뒤는 공백 없이 비웁니다.
            - 물체와 개수와 목적지의 순서는 등장 순서를 따릅니다.

            <특수 규칙>
            - 명확한 물체 명칭과 갖다놓을 위치가 없지만 문맥상 유추 가능한 경우(예: "병든 거" → rotten)는 리스트 내 항목으로 최대한 추론해 반환하세요.
            - 다수의 도구와 목적지가 동시에 등장할 경우 각각에 대해 정확히 매칭하여 순서대로 출력하세요.

            <예시>
            - 입력: "pimang을 pos1에 가져다 놔"  
            출력: pimang / pos1

            - 입력: "pimang 2개를 pos1에 가져다놔"
            출력: pimang 2 pos1

            - 입력: "병든 피망을 pos2에 넣어줘"  
            출력: rotten / pos2

            - 입력: "병든 거만 버려줘"  
            출력: rotten / pos2

            - 입력: "수확해줘"  
            출력: pimang / pos1 pos2
            
            - 입력: "병든pimang과 pimang 수확해줘"  
            출력: pimang rotten / pos1 pos2

            - 입력: "pimang은 pos1에 두고 병든 pimang은 pos2에 둬"  
            출력: pimang rotten / pos1 pos2

            <사용자 입력>
            {user_input}                                                 
        """

        self.prompt_template = PromptTemplate(
            input_variables=["user_input"], template=prompt_content
        )
        self.lang_chain = LLMChain(llm=self.llm, prompt=self.prompt_template)
        self.stt = STT(openai_api_key=openai_api_key)


        super().__init__("get_keyword_node")
        # 오디오 설정
        mic_config = MicConfig(
            chunk=12000,
            rate=48000,
            channels=1,
            record_seconds=5,
            fmt=pyaudio.paInt16,
            device_index=10,
            buffer_size=24000,
        )
        self.mic_controller = MicController(config=mic_config)
        # self.ai_processor = AIProcessor()

        self.get_logger().info("MicRecorderNode initialized.")
        self.get_logger().info("wait for client's request...")
        self.get_keyword_srv = self.create_service(
            Trigger, "get_keyword", self.get_keyword
        )
        self.wakeup_word = WakeupWord(mic_config.buffer_size)

    def extract_keyword(self, output_message):
        response = self.lang_chain.invoke({"user_input": output_message})
        result = response["text"].strip()

        # 기본값 초기화
        object, num, target = [], [], []

        try:
            parts = result.split("/")
            if len(parts) == 3:
                object = parts[0].strip().split() if parts[0].strip() else []
                num = parts[1].strip().split() if parts[1].strip() else []
                target = parts[2].strip().split() if parts[2].strip() else []
            elif len(parts) == 2:
                object = parts[0].strip().split() if parts[0].strip() else []
                num = []  # 중간 num 생략된 경우
                target = parts[1].strip().split() if parts[1].strip() else []
            elif len(parts) == 1:
                object = parts[0].strip().split() if parts[0].strip() else []
                num = []
                target = []
            else:
                self.get_logger().warn("LLM 응답 형식이 예상과 다릅니다.")
        except Exception as e:
            self.get_logger().error(f"키워드 추출 중 오류 발생: {str(e)}")

        print(f"llm's response: {result}")
        print(f"object: {object}")
        print(f"num: {num}")
        print(f"target: {target}")

        return object, num, target


    def get_keyword(self, request, response):
        try:
            print("open stream")
            self.mic_controller.open_stream()
            self.wakeup_word.set_stream(self.mic_controller.stream)
        except OSError:
            self.get_logger().error("Error: Failed to open audio stream")
            self.get_logger().error("please check your device index")
            return None

        while not self.wakeup_word.is_wakeup():
            pass

        output_message = self.stt.speech2text()
        object_list, num_list, target_list = self.extract_keyword(output_message)

        self.get_logger().warn(f"Detected tools: {object_list}, numbers: {num_list}, targets: {target_list}")

        response.success = True
        response.message = f"{' '.join(object_list)} / {' '.join(num_list)} / {' '.join(target_list)}"
        return response


def main():
    rclpy.init()
    node = GetKeyword()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

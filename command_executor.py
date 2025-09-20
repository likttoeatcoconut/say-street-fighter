import time
import keyboard
from queue import Empty
from threading import Event
from multiprocessing import Queue
from config import COMMAND_MAPPING, PLAYER_CONFIG

class CommandExecutor:
    def __init__(self, command_queue: Queue, stop_event: Event):
        self.command_queue = command_queue
        self.stop_event = stop_event
        self.location = PLAYER_CONFIG['LOCATION']
        self.zhen_s = 0.017 #每帧秒数

    def start(self):
        """启动指令执行线程"""
        print("指令执行线程启动")

        while not self.stop_event.is_set():
            try:
                command = self.command_queue.get(timeout=0.1)
                print(f"执行指令: {command}")
                start_time = time.time()
                self.execute_command(command)
                execution_time = time.time() - start_time
                print(f"执行用时: {execution_time:.4f}秒")
            except Empty:
                continue
            except Exception as e:
                print(f"指令执行错误: {e}")

    def execute_command(self, command):
        """执行指令"""
        if command not in COMMAND_MAPPING:
            return
        
        keys = COMMAND_MAPPING[command]
        for key in keys:
            if isinstance(key, tuple):
                # 处理同时按键
                for k in key:
                    self.press(k)
                time.sleep(self.zhen_s)
                for k in key:
                    self.release(k)
            elif key:
                if key.startswith("#"):
                    # 处理放帧
                    num = int(key.split("#")[1])
                    time.sleep(1.0 / 60 * num)
                elif key.startswith("$"):
                    # 处理嵌套指令
                    cmd = key.split("$")[1]
                    self.execute_command(cmd)
                elif key == "@":
                    self.location = 'right' if self.location == 'left' else 'left'
                else:
                    # 处理单个按键
                    self.press(key)
                    time.sleep(self.zhen_s)
                    self.release(key)
            else:
                time.sleep(self.zhen_s)

    def press(self, key):
        """按下按键"""
        keyboard.press(self.key_map_by_location(key))

    def release(self, key):
        """释放按键"""
        keyboard.release(self.key_map_by_location(key))

    def key_map_by_location(self, key):
        """根据位置映射按键"""
        if self.location == 'right':
            # ad互换
            if key == 'd':
                return 'a'
            elif key == 'a':
                return 'd'
        return key
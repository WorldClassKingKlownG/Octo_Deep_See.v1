# src/bot_system/enhanced_orchestrator.py
from enum import Enum
import asyncio
from bot_system.enhanced_bot_structure import Agent
from deepseek_model import setup_model

class BotRole(Enum):
    FILE_MANAGER = "file_manager"
    NETWORK_CONTROLLER = "network_controller"
    UI_NAVIGATOR = "ui_navigator"
    DATA_PROCESSOR = "data_processor"
    SECURITY_MONITOR = "security_monitor"
    RESOURCE_OPTIMIZER = "resource_optimizer"
    COMMUNICATION_HANDLER = "communication_handler"
    TASK_COORDINATOR = "task_coordinator"

class AutonomousBot:
    def __init__(self, role: BotRole, model, tokenizer):
        self.role = role
        self.model = model
        self.tokenizer = tokenizer
        self.active = True
        self.task_queue = asyncio.Queue()

    async def execute_task(self, task):
        prompt = f"As a {self.role.value} bot, perform: {task}"
        inputs = self.tokenizer(prompt, return_tensors="pt")
        response = self.model.generate(**inputs)
        return self.tokenizer.decode(response[0])

    async def run(self):
        while self.active:
            task = await self.task_queue.get()
            result = await self.execute_task(task)
            await self.report_result(result)

    async def report_result(self, result):
        print(f"Bot {self.role.value} completed the task with result: {result}")

class BotOrchestrator:
    def __init__(self, model, tokenizer):
        self.bots = {
            BotRole.FILE_MANAGER: AutonomousBot(BotRole.FILE_MANAGER, model, tokenizer),
            BotRole.NETWORK_CONTROLLER: AutonomousBot(BotRole.NETWORK_CONTROLLER, model, tokenizer),
            BotRole.UI_NAVIGATOR: AutonomousBot(BotRole.UI_NAVIGATOR, model, tokenizer),
            BotRole.DATA_PROCESSOR: AutonomousBot(BotRole.DATA_PROCESSOR, model, tokenizer),
            BotRole.SECURITY_MONITOR: AutonomousBot(BotRole.SECURITY_MONITOR, model, tokenizer),
            BotRole.RESOURCE_OPTIMIZER: AutonomousBot(BotRole.RESOURCE_OPTIMIZER, model, tokenizer),
            BotRole.COMMUNICATION_HANDLER: AutonomousBot(BotRole.COMMUNICATION_HANDLER, model, tokenizer),
            BotRole.TASK_COORDINATOR: AutonomousBot(BotRole.TASK_COORDINATOR, model, tokenizer)
        }

    async def distribute_task(self, task, role):
        await self.bots[role].task_queue.put(task)

    async def start_all_bots(self):
        bot_tasks = [bot.run() for bot in self.bots.values()]
        await asyncio.gather(*bot_tasks)

    async def stop_all_bots(self):
        for bot in self.bots.values():
            bot.active = False

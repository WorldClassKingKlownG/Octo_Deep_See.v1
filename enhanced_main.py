# src/enhanced_main.py
import asyncio
from deepseek_model import setup_model
from bot_system.task_distribution import train_model, TextDataset
from bot_system.enhanced_bot_structure import Agent, collect_training_data, GRID_SIZE, FOOD
from bot_system.enhanced_orchestrator import BotRole, BotOrchestrator
from bot_system.phone_bots import PhoneFileManagerBot, PhoneNetworkBot, PhoneUIBot, PhoneDataBot

class EnhancedDeepSeekSystem:
    def __init__(self):
        self.model, self.tokenizer = setup_model()
        self.bot_orchestrator = BotOrchestrator(self.model, self.tokenizer)
        
    async def initialize(self):
        print("Starting bot system...")
        await self.bot_orchestrator.start_all_bots()
        
    async def execute_phone_task(self, task_description):
        # Analyze task and delegate to appropriate bot
        task_analysis = await self.analyze_task(task_description)
        role = self.determine_bot_role(task_analysis)
        await self.bot_orchestrator.distribute_task(task_description, role)
        
    async def analyze_task(self, task):
        prompt = f"Analyze the following task and determine required actions: {task}"
        inputs = self.tokenizer(prompt, return_tensors="pt")
        analysis = self.model.generate(**inputs)
        return self.tokenizer.decode(analysis[0])

    def determine_bot_role(self, task_analysis):
        # Dummy implementation for task analysis to bot role mapping
        if "file" in task_analysis:
            return BotRole.FILE_MANAGER
        elif "network" in task_analysis:
            return BotRole.NETWORK_CONTROLLER
        elif "UI" in task_analysis:
            return BotRole.UI_NAVIGATOR
        elif "data" in task_analysis:
            return BotRole.DATA_PROCESSOR
        else:
            return BotRole.TASK_COORDINATOR

if __name__ == "__main__":
    system = EnhancedDeepSeekSystem()
    asyncio.run(system.initialize())

    sample_task = "Process user data for analysis"
    asyncio.run(system.execute_phone_task(sample_task))

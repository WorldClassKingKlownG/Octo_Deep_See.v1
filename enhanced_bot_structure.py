# src/bot_system/enhanced_bot_structure.py
import numpy as np
import random

GRID_SIZE = 20
EMPTY = 0
FOOD = 1
AGENT = 2

class Agent:
    def __init__(self, x, y, model, tokenizer):
        self.x = x
        self.y = y
        self.energy = 100
        self.model = model
        self.tokenizer = tokenizer

    def perceive(self, grid):
        surroundings = [
            grid[(self.x - 1) % GRID_SIZE, self.y],  # Up
            grid[(self.x + 1) % GRID_SIZE, self.y],  # Down
            grid[self.x, (self.y - 1) % GRID_SIZE],  # Left
            grid[self.x, (self.y + 1) % GRID_SIZE],  # Right
            grid[self.x, self.y]  # Center
        ]
        return surroundings

    def generate_prompt(self, surroundings):
        return f"Surroundings: {' '.join(map(str, surroundings))}. Objective: survival and procreation. Action:"

    def decide(self, surroundings):
        prompt = self.generate_prompt(surroundings)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(
            inputs.input_ids,
            max_length=50,
            num_return_sequences=1,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )
        action_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True).split("Action:")[-1].strip()
        action_map = {"up": 0, "down": 1, "left": 2, "right": 3}
        return action_map.get(action_text, random.choice([0, 1, 2, 3]))

    def act(self, action):
        if action == 0:  # Move up
            self.x = (self.x - 1) % GRID_SIZE
        elif action == 1:  # Move down
            self.x = (self.x + 1) % GRID_SIZE
        elif action == 2:  # Move left
            self.y = (self.y - 1) % GRID_SIZE
        elif action == 3:  # Move right
            self.y = (self.y + 1) % GRID_SIZE

    def update(self, grid):
        surroundings = self.perceive(grid)
        action = self.decide(surroundings)
        self.act(action)
        self.energy -= 1
        if grid[self.x, self.y] == FOOD:
            self.energy += 10
            grid[self.x, self.y] = EMPTY
        if self.energy >= 200:
            self.energy = 100
            return Agent(self.x, self.y, self.model, self.tokenizer)
        return None

    def generate_training_data(self, grid):
        surroundings = self.perceive(grid)
        action = self.decide(surroundings)
        prompt = self.generate_prompt(surroundings)
        action_map_reverse = {0: "up", 1: "down", 2: "left", 3: "right"}
        action_text = action_map_reverse[action]
        return {"text": prompt + action_text}

def collect_training_data(agents, grid, steps=100):
    training_data = []
    for step in range(steps):
        for agent in agents:
            training_data.append(agent.generate_training_data(grid))
            agent.update(grid)
    return training_data

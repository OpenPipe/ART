"""
Skill-RL: Train models to follow markdown skill instructions.

This module provides the core training loop and evaluation for
teaching small language models to execute structured skills.
"""

import json
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import art
from art.rewards import RulerReward


@dataclass
class SkillConfig:
    """Configuration for a skill training run."""
    skill_path: str
    tasks_path: str
    eval_path: Optional[str] = None
    base_model: str = "Qwen/Qwen2.5-3B-Instruct"
    project: str = "skill-rl"
    epochs: int = 10
    

def load_skill(skill_path: str) -> str:
    """Load a skill definition from markdown file."""
    return Path(skill_path).read_text()


def load_tasks(tasks_path: str) -> list[dict]:
    """Load training tasks from JSONL file."""
    tasks = []
    with open(tasks_path) as f:
        for line in f:
            if line.strip():
                tasks.append(json.loads(line))
    return tasks


def build_prompt(skill_md: str, task: str) -> str:
    """Build the prompt for the model."""
    return f"""You have the following skill loaded:

<skill>
{skill_md}
</skill>

---

User request: {task}

Follow the skill instructions exactly. Output only the result, no explanation."""


def create_reward_fn(skill_md: str):
    """Create a reward function for evaluating skill adherence."""
    
    def skill_reward(task: str, response: str) -> float:
        """
        Evaluate how well the response follows the skill.
        
        Returns a score from 0.0 to 1.0 based on:
        - Format compliance (30%)
        - Procedure adherence (40%)
        - Correctness (30%)
        """
        ruler = RulerReward(
            criteria=f"""
Evaluate this response against the skill instructions.

SKILL DEFINITION:
{skill_md}

USER TASK:
{task}

MODEL RESPONSE:
{response}

Score each dimension from 0-100:

1. FORMAT (weight 0.3): Does the output match the exact format specified in the skill?
   - Check all format requirements in the skill
   - Penalize missing sections, wrong structure, incorrect delimiters
   
2. PROCEDURE (weight 0.4): Were all steps in the procedure followed?
   - Check each numbered step in the skill
   - Verify the order was correct
   - Penalize skipped or out-of-order steps

3. CORRECTNESS (weight 0.3): Is the output actually correct and useful?
   - Would a human accept this output?
   - Does it make sense for the given task?

Return a JSON object with scores and brief justifications:
{{"format": {{"score": N, "reason": "..."}}, "procedure": {{"score": N, "reason": "..."}}, "correctness": {{"score": N, "reason": "..."}}}}
"""
        )
        
        result = ruler.evaluate(response)
        
        # Parse the structured response
        try:
            scores = json.loads(result)
            total = (
                scores["format"]["score"] * 0.3 +
                scores["procedure"]["score"] * 0.4 +
                scores["correctness"]["score"] * 0.3
            )
            return total / 100.0
        except (json.JSONDecodeError, KeyError):
            # Fallback to simple scoring if structured parsing fails
            return ruler.score(response) / 100.0
    
    return skill_reward


def train_skill(config: SkillConfig) -> art.TrainableModel:
    """
    Train a model to execute a skill.
    
    Args:
        config: Training configuration
        
    Returns:
        Trained model
    """
    # Load skill and tasks
    skill_md = load_skill(config.skill_path)
    tasks = load_tasks(config.tasks_path)
    
    # Create reward function
    reward_fn = create_reward_fn(skill_md)
    
    # Initialize model
    skill_name = Path(config.skill_path).parent.name
    model = art.TrainableModel(
        project=config.project,
        name=f"{skill_name}-specialist",
        base_model=config.base_model
    )
    
    print(f"Training {config.base_model} on skill: {skill_name}")
    print(f"Tasks: {len(tasks)}, Epochs: {config.epochs}")
    
    # Training loop
    for epoch in range(config.epochs):
        epoch_rewards = []
        
        for task_data in tasks:
            task = task_data["task"]
            
            # Build prompt and generate
            prompt = build_prompt(skill_md, task)
            response = model.generate(prompt)
            
            # Calculate reward
            reward = reward_fn(task, response)
            epoch_rewards.append(reward)
            
            # Train step
            model.train_step(prompt, response, reward)
        
        avg_reward = sum(epoch_rewards) / len(epoch_rewards)
        print(f"Epoch {epoch + 1}/{config.epochs}: avg_reward={avg_reward:.3f}")
    
    return model


def evaluate_skill(
    model: art.TrainableModel,
    skill_path: str,
    eval_path: str
) -> dict:
    """
    Evaluate a trained model on held-out tasks.
    
    Returns:
        Dictionary with evaluation metrics
    """
    skill_md = load_skill(skill_path)
    eval_tasks = load_tasks(eval_path)
    reward_fn = create_reward_fn(skill_md)
    
    results = []
    for task_data in eval_tasks:
        task = task_data["task"]
        prompt = build_prompt(skill_md, task)
        response = model.generate(prompt)
        reward = reward_fn(task, response)
        
        results.append({
            "task": task,
            "response": response,
            "reward": reward
        })
    
    avg_score = sum(r["reward"] for r in results) / len(results)
    
    return {
        "average_score": avg_score,
        "num_tasks": len(results),
        "results": results
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train a model on a skill")
    parser.add_argument("--skill", required=True, help="Path to SKILL.md")
    parser.add_argument("--tasks", help="Path to tasks.jsonl (default: same dir as skill)")
    parser.add_argument("--eval", help="Path to eval.jsonl for evaluation")
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct", help="Base model")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs")
    parser.add_argument("--output", help="Output directory for checkpoints")
    
    args = parser.parse_args()
    
    # Default tasks path
    skill_dir = Path(args.skill).parent
    tasks_path = args.tasks or str(skill_dir / "tasks.jsonl")
    
    config = SkillConfig(
        skill_path=args.skill,
        tasks_path=tasks_path,
        eval_path=args.eval,
        base_model=args.model,
        epochs=args.epochs
    )
    
    trained_model = train_skill(config)
    
    # Evaluate if eval set provided
    if args.eval:
        print("\nEvaluating on held-out tasks...")
        eval_results = evaluate_skill(trained_model, args.skill, args.eval)
        print(f"Evaluation score: {eval_results['average_score']:.3f}")
    
    # Save checkpoint
    if args.output:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        trained_model.save(output_dir / "model")
        print(f"Model saved to {output_dir / 'model'}")

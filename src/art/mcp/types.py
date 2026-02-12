from collections import Counter
from dataclasses import dataclass
import json
import random
from typing import Any, Dict, List, Optional

from art.utils.logging import _C, dim, info


@dataclass
class MCPTool:
    """Represents an MCP tool."""

    name: str
    description: str
    parameters: Dict[str, Any]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MCPTool":
        """Create a Tool from a dictionary."""
        return cls(
            name=data.get("name", ""),
            description=data.get("description", ""),
            parameters=data.get("parameters", {}),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert the tool to a dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }

    def to_tool_schema(self) -> Dict[str, Any]:
        """Convert the tool to a tool schema."""
        return {
            "type": "function",
            "function": self.to_dict(),
        }


@dataclass
class MCPResource:
    """Represents an MCP resource."""

    uri: str
    name: str
    description: str
    mime_type: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MCPResource":
        """Create a Resource from a dictionary."""
        return cls(
            uri=data.get("uri", ""),
            name=data.get("name", ""),
            description=data.get("description", ""),
            mime_type=data.get("mimeType") or data.get("mime_type"),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert the resource to a dictionary."""
        result = {"uri": self.uri, "name": self.name, "description": self.description}
        if self.mime_type:
            result["mimeType"] = self.mime_type
        return result


@dataclass
class GeneratedScenario:
    """A single scenario for testing AI agents."""

    task: str
    difficulty: int

    def __post_init__(self):
        if not isinstance(self.difficulty, int) or not 1 <= self.difficulty <= 5:
            raise ValueError("Difficulty must be an integer between 1 and 5")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GeneratedScenario":
        """Create a GeneratedScenario from a dictionary."""
        return cls(task=data["task"], difficulty=data["difficulty"])

    def to_dict(self) -> Dict[str, Any]:
        """Convert the scenario to a dictionary."""
        return {"task": self.task, "difficulty": self.difficulty}

    def preview(self, max_length: int = 120) -> str:
        """Get a preview of the scenario task."""
        if len(self.task) <= max_length:
            return self.task
        return self.task[:max_length].strip() + "…"


class GeneratedScenarioCollection:
    """A collection of scenarios with utilities for management and analysis."""

    def __init__(self, scenarios: List[GeneratedScenario]):
        self.scenarios = scenarios

    @classmethod
    def from_dicts(cls, data: List[Dict[str, Any]]) -> "GeneratedScenarioCollection":
        """Create a GeneratedScenarioCollection from a list of dictionaries."""
        scenarios = [GeneratedScenario.from_dict(item) for item in data]
        return cls(scenarios)

    @classmethod
    def from_json(cls, json_str: str) -> "GeneratedScenarioCollection":
        """Create a GeneratedScenarioCollection from a JSON string."""
        data = json.loads(json_str)
        if "scenarios" in data:
            scenarios_data = data["scenarios"]
        else:
            scenarios_data = data if isinstance(data, list) else list(data.values())[0]
        return cls.from_dicts(scenarios_data)

    def to_dicts(self) -> List[Dict[str, Any]]:
        """Convert all scenarios to dictionaries."""
        return [scenario.to_dict() for scenario in self.scenarios]

    def to_json(self, indent: Optional[int] = None) -> str:
        """Convert the collection to JSON."""
        return json.dumps({"scenarios": self.to_dicts()}, indent=indent)

    def __len__(self) -> int:
        return len(self.scenarios)

    def __iter__(self):
        return iter(self.scenarios)

    def __getitem__(self, index):
        return self.scenarios[index]

    def shuffle(self) -> "GeneratedScenarioCollection":
        """Return a new collection with shuffled scenarios."""
        shuffled = self.scenarios.copy()
        random.shuffle(shuffled)
        return GeneratedScenarioCollection(shuffled)

    def split(
        self, train_size: int
    ) -> tuple["GeneratedScenarioCollection", "GeneratedScenarioCollection"]:
        """Split the collection into train and validation sets."""
        if train_size > len(self.scenarios):
            raise ValueError(
                f"train_size ({train_size}) cannot be larger than total scenarios ({len(self.scenarios)})"
            )

        train_scenarios = self.scenarios[:train_size]
        val_scenarios = self.scenarios[train_size:]

        return GeneratedScenarioCollection(
            train_scenarios
        ), GeneratedScenarioCollection(val_scenarios)

    def filter_by_difficulty(
        self, min_difficulty: int = 1, max_difficulty: int = 5
    ) -> "GeneratedScenarioCollection":
        """Filter scenarios by difficulty range."""
        filtered = [
            scenario
            for scenario in self.scenarios
            if min_difficulty <= scenario.difficulty <= max_difficulty
        ]
        return GeneratedScenarioCollection(filtered)

    def get_difficulty_distribution(self) -> Counter:
        """Get the distribution of difficulties."""
        return Counter(scenario.difficulty for scenario in self.scenarios)

    def preview(self, n: int = 5, max_task_length: int = 120) -> None:
        """Preview the first n scenarios."""
        n = min(n, len(self.scenarios))
        for i in range(n):
            scenario = self.scenarios[i]
            preview_text = scenario.preview(max_task_length)
            dim(
                f"   {i + 1}. {preview_text}  "
                f"{_C.GRAY}(difficulty {scenario.difficulty}/5){_C.RESET}"
            )

    def print_difficulty_distribution(self) -> None:
        """Print a visual representation of the difficulty distribution."""
        diff_counts = self.get_difficulty_distribution()
        info("Difficulty distribution:")
        for d in range(1, 6):
            cnt = diff_counts.get(d, 0)
            bar = "█" * min(cnt, 30)
            dim(f"   {d}/5: {cnt:3d}  {bar}")

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the scenario collection."""
        return {
            "total_scenarios": len(self.scenarios),
            "difficulty_distribution": dict(self.get_difficulty_distribution()),
            "avg_difficulty": sum(s.difficulty for s in self.scenarios)
            / len(self.scenarios)
            if self.scenarios
            else 0,
            "avg_task_length": sum(len(s.task) for s in self.scenarios)
            / len(self.scenarios)
            if self.scenarios
            else 0,
        }

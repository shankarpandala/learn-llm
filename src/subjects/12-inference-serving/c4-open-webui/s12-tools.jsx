import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function Tools() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Function Calling & Tools</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        Open WebUI supports tools and functions that extend the LLM's capabilities beyond text
        generation. Tools let the model execute code, query APIs, access databases, and interact
        with external systems, all within the chat interface.
      </p>

      <DefinitionBlock
        title="Tools in Open WebUI"
        definition="Tools are Python functions that the LLM can invoke during a conversation. Open WebUI provides a framework for defining tools with descriptions, parameter schemas, and execution logic. The model decides when and how to use available tools based on the user's request."
        id="def-tools"
      />

      <ExampleBlock
        title="Built-in vs Custom Tools"
        problem="What tools come with Open WebUI and how do you add your own?"
        steps={[
          { formula: 'Built-in: web search, code execution, image generation', explanation: 'Available out of the box with minimal configuration.' },
          { formula: 'Community: browse and install from the tool marketplace', explanation: 'Open WebUI has a community repository of shared tools.' },
          { formula: 'Custom: write Python functions with the Tools API', explanation: 'Define your own tools for any custom integration.' },
        ]}
        id="example-tools"
      />

      <PythonCode
        title="custom_tool.py"
        code={`# Example: Custom tool for Open WebUI
# This goes in the Open WebUI Tools editor (Admin > Workspace > Tools)

"""
title: Weather Tool
description: Get current weather for a location
author: your-name
version: 0.1.0
"""

import requests
from pydantic import BaseModel, Field


class Tools:
    class Valves(BaseModel):
        api_key: str = Field(
            default="", description="OpenWeatherMap API key"
        )

    def __init__(self):
        self.valves = self.Valves()

    def get_weather(
        self,
        location: str,
    ) -> str:
        """
        Get the current weather for a given location.

        :param location: City name (e.g., 'London', 'New York')
        :return: Weather description with temperature
        """
        try:
            resp = requests.get(
                "https://api.openweathermap.org/data/2.5/weather",
                params={
                    "q": location,
                    "appid": self.valves.api_key,
                    "units": "metric",
                },
                timeout=10,
            )
            data = resp.json()
            temp = data["main"]["temp"]
            desc = data["weather"][0]["description"]
            return f"Weather in {location}: {desc}, {temp}°C"
        except Exception as e:
            return f"Could not get weather: {str(e)}"`}
        id="code-custom-tool"
      />

      <PythonCode
        title="calculator_tool.py"
        code={`# Another example: Calculator tool
# Paste into Tools editor in Open WebUI

"""
title: Calculator
description: Perform mathematical calculations
author: your-name
version: 0.1.0
"""

import math
from pydantic import BaseModel


class Tools:
    def calculate(self, expression: str) -> str:
        """
        Evaluate a mathematical expression safely.

        :param expression: Math expression to evaluate (e.g., 'sqrt(144) + 2**3')
        :return: The result of the calculation
        """
        # Safe math functions
        safe_dict = {
            "abs": abs, "round": round, "min": min, "max": max,
            "sum": sum, "pow": pow, "sqrt": math.sqrt,
            "sin": math.sin, "cos": math.cos, "tan": math.tan,
            "log": math.log, "log10": math.log10, "pi": math.pi,
            "e": math.e, "inf": float("inf"),
        }
        try:
            result = eval(expression, {"__builtins__": {}}, safe_dict)
            return f"Result: {expression} = {result}"
        except Exception as e:
            return f"Error evaluating '{expression}': {str(e)}"

    def unit_convert(self, value: float, from_unit: str, to_unit: str) -> str:
        """
        Convert between common units.

        :param value: Numeric value to convert
        :param from_unit: Source unit (e.g., 'km', 'miles', 'celsius')
        :param to_unit: Target unit (e.g., 'miles', 'km', 'fahrenheit')
        :return: Converted value with units
        """
        conversions = {
            ("km", "miles"): lambda v: v * 0.621371,
            ("miles", "km"): lambda v: v * 1.60934,
            ("celsius", "fahrenheit"): lambda v: v * 9/5 + 32,
            ("fahrenheit", "celsius"): lambda v: (v - 32) * 5/9,
            ("kg", "lbs"): lambda v: v * 2.20462,
            ("lbs", "kg"): lambda v: v * 0.453592,
        }
        key = (from_unit.lower(), to_unit.lower())
        if key in conversions:
            result = conversions[key](value)
            return f"{value} {from_unit} = {result:.4f} {to_unit}"
        return f"Unknown conversion: {from_unit} -> {to_unit}"`}
        id="code-calculator"
      />

      <NoteBlock
        type="tip"
        title="Tool Discovery"
        content="Browse community tools at openwebui.com. Tools can be installed with one click and include integrations for Jira, Slack, databases, file systems, and more. Check the tool's source code before installing to understand what it does."
        id="note-discovery"
      />

      <WarningBlock
        title="Tool Security"
        content="Custom tools execute arbitrary Python code on your server. Only install tools from trusted sources and review the code before enabling. Malicious tools could access your file system, network, or other resources. Admins should control which tools are available to users."
        id="warning-security"
      />
    </div>
  )
}

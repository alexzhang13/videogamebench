import asyncio
import base64
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from PIL import Image
from io import BytesIO
import re
import threading
from src.utils import is_same_image
from src.emulators.dos.browser_controller import BrowserController
from src.llm.llm_client import LLMClient
from src.llm.prompts import SYSTEM_PROMPTS, TASK_PROMPTS, GBA_PROMPT, REFLECTION_PROMPT, GBA_REALTIME_PROMPT
from src.llm.utils import parse_actions_response, convert_to_dict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Message:
    """Container for a message in the conversation history."""
    
    def __init__(self, role: str, content: Any, has_image: bool = False, tokens: int = 0):
        self.role = role
        self.content = content
        self.has_image = has_image
        self.tokens = tokens  # Approximate token count
        self.timestamp = datetime.now()
        
    def __str__(self):
        if isinstance(self.content, str):
            preview = self.content[:50] + "..." if len(self.content) > 50 else self.content
            return f"{self.role}: {preview}"
        else:
            return f"{self.role}: [Complex content with image={self.has_image}]"

class VideoGameBenchAgent:
    """
    Base class for all VG agents.
    """
    def __init__(self, 
                 model: str, 
                 api_key: str, 
                 game: str, 
                 task_prompt: str,
                 headless: bool = False, 
                 temperature: float = 0.7, 
                 max_tokens: int = 1024, 
                 max_history_tokens: int = 4000, 
                 context_window: int = 10, 
                 log_dir: Optional[Path] = None,
                 enable_ui: bool = False,
                 api_base: Optional[str] = None):
        # Set up logging directory
        if log_dir is None:
            # Remove .,: for compatability with Windows file systems
            model_name = model.replace("/", "-").replace(".", "-").replace(":", "-")
            self.log_dir = Path("logs") / f"{game.lower()}" / model_name / datetime.now().strftime("%Y%m%d_%H%M%S")
            self.log_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.log_dir = log_dir

        # Set up file logger
        self.file_logger = self._setup_file_logger()
        self.file_logger.info(f"Initializing {self.__class__.__name__} with model: {model}")

        # Initialize LLM client
        self.llm_client = LLMClient(
            model=model,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            log_dir=self.log_dir / "llm",
            api_base=api_base
        )

        # Common attributes
        self.max_history_tokens = max_history_tokens
        self.context_window = context_window
        self.headless = headless # TODO: make this do something
        self.game = game
        self.task_prompt = task_prompt
        self.system_instruction_prompt = SYSTEM_PROMPTS.get(game, "")
        self.system_prompt = {
            "role": "system",
            "content": f"{self.system_instruction_prompt}\n\n{self.task_prompt}"
        }
        self.context_history: List[Message] = []
        self.full_history: List[Dict[str, Any]] = []
        
        self.reflection_memory = ""
        self.step_count = 0
        
        # Create consolidated log files
        self.reflection_log_file = self.log_dir / "reflections.txt"
        self.reflection_log_file.touch()

        logger.info(f"{self.__class__.__name__} initialized. Logging to: {self.log_dir}")

        # Add checkpoint tracking
        self.current_checkpoint = 0
        self.total_checkpoints = 0
        
        # Initialize UI if enabled
        self.ui = None
        if enable_ui:
            from src.ui.vgagent_monitor import AgentMonitorUI
            self.ui = AgentMonitorUI(f"{model} agent playing {self.game} on VideoGameBench")

    
    def add_to_history(self, role: str, content: Any, has_image: bool = False, tokens: int = 0) -> None:
        """Add a message to both full history and context history."""
        message = Message(role, content, has_image, tokens)
        
        # Add to context history (with images)
        self.context_history.append(message)
        if len(self.context_history) > self.context_window:
            self.context_history = self.context_history[-self.context_window:]
            self.file_logger.info(f"Pruned conversation history to {self.context_window} messages")
        
        # Add to full history (without images)
        if isinstance(content, list):
            # Extract text content from multimodal messages
            text_content = ""
            for item in content:
                if item.get("type") == "text":
                    text_content += item.get("text", "") + " "
        else:
            text_content = content
            
        self.full_history.append({
            "role": role,
            "content": text_content.strip(),
            "timestamp": datetime.now(),
            "tokens": tokens
        })
        
        # Log the exchange
        if role == "user":
            log_msg = f"Step {self.step_count} - User observation"
            if has_image:
                log_msg += " with image"
            self.file_logger.info(log_msg)
        else:
            self.file_logger.info(f"Step {self.step_count} - AI action: {text_content}")
        
        self._prune_history()


    def _prune_history(self) -> None:
        """Remove oldest messages if token count exceeds max_history_tokens."""
        # Calculate total tokens
        total_tokens = sum(msg.tokens for msg in self.context_history)
        
        # Count number of images in history
        image_count = sum(1 for msg in self.context_history if msg.has_image)
        
        # Keep removing oldest messages until we're under all limits
        while (total_tokens > self.max_history_tokens or 
               len(self.context_history) > self.context_window):
            
            # If we have too many images, remove the oldest image and its text
            if image_count > self.context_window // 2 and self.context_history[0].has_image:
                # Remove the image message
                removed = self.context_history.pop(0)
                total_tokens -= removed.tokens
                image_count -= 1
                
                # Also remove the text message before it if it exists
                if self.context_history and not self.context_history[0].has_image:
                    removed = self.context_history.pop(0)
                    total_tokens -= removed.tokens
            
            # Otherwise just remove oldest message
            else:
                removed = self.context_history.pop(0)
                total_tokens -= removed.tokens
                if removed.has_image:
                    image_count -= 1
                    
            logger.debug(f"Pruned message from history: {removed}")

    def _setup_file_logger(self) -> logging.Logger:
        """Set up a file logger for this session."""
        file_logger = logging.getLogger(f"{self.__class__.__name__.lower()}_{id(self)}")
        file_logger.setLevel(logging.INFO)
        
        # Create file handler
        log_file = self.log_dir / "agent_session.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        
        # Add handler to logger
        file_logger.addHandler(file_handler)
        
        return file_logger

    def save_reflection(self) -> None:
        """Save reflection to consolidated file."""
        if self.reflection_memory:
            with open(self.reflection_log_file, "a") as f:
                f.write(f"\n\n-=-=-=-=-=-reflection_{self.step_count}-=-=-=-=-=\n\n")
                f.write(self.reflection_memory)
            self.file_logger.info(f"Updated reflection memory saved to {self.reflection_log_file}")

        if self.ui:
            print("Updating reflection memory in UI...")
            self.ui.update_reflection_memory(self.reflection_memory)
    
    def update_steps_count(self, count: Optional[int] = None) -> None:
        """Update the steps counter display."""
        if self.ui:
            self.ui.update_steps_count(count)

    def setup_checkpoints(self, num_checkpoints: int):
        """Initialize checkpoint tracking."""
        self.total_checkpoints = num_checkpoints
        self.current_checkpoint = 0
        if self.ui:
            print("Setting up checkpoints in UI... with num_checkpoints:", num_checkpoints)
            self.ui.setup_checkpoints(num_checkpoints)

    def update_checkpoint(self, checkpoint_idx: int):
        """Update the current checkpoint progress."""
        if checkpoint_idx > self.current_checkpoint:
            self.current_checkpoint = checkpoint_idx
            if self.ui:
                # Mark all checkpoints up to current as completed
                for idx in range(1, checkpoint_idx+1):
                    print(f"Agent: Updating checkpoint {idx} to completed")
                    self.ui.update_checkpoint(idx, completed=True)

class GameBoyVGAgent(VideoGameBenchAgent):
    """
    VideoGameBench Agent that uses the ReACT method to play a Game Boy game.
    Unlike the web browsing agent, this agent does not hold the browser / game object.
    """
    def __init__(
        self,
        model: str,
        api_key: str,
        game: str,
        task_prompt: str,
        headless: bool = False,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        max_history_tokens: int = 4000,
        context_window: int = 10,
        log_dir: Optional[Path] = None,
        realtime: bool = False,
        enable_ui: bool = False,
        api_base: Optional[str] = None
    ):
        """
        Initialize the GBA agent.
        
        Args:
            model: The LLM model to use
            api_key: The API key for the model provider
            game_type: The type of game being played
            headless: Whether to run the browser in headless mode
            temperature: The temperature for LLM generation
            max_tokens: The maximum number of tokens to generate
            max_history_tokens: The maximum number of tokens to keep in history
            log_dir: Optional custom log directory path
        """
        super().__init__(
            model=model,
            api_key=api_key,
            game=game,
            headless=headless,
            temperature=temperature,
            max_tokens=max_tokens,
            max_history_tokens=max_history_tokens,
            context_window=context_window,
            log_dir=log_dir,
            enable_ui=enable_ui,
            task_prompt=task_prompt,
            api_base=api_base
        )
        
        self.realtime = realtime
        self.system_instruction_prompt = SYSTEM_PROMPTS["gba_realtime"] if realtime else SYSTEM_PROMPTS["gba"]
        self.system_prompt["content"] = f"{self.system_instruction_prompt}\n\n{self.task_prompt}"
        self.gba_prompt = GBA_REALTIME_PROMPT if realtime else GBA_PROMPT
        
        self.image_dir = None
        self.context_window = context_window

        self.action = ""
        self.prev_action = ""

        self.image_dir = self.log_dir / "game_screen"
        self.image_dir.mkdir(exist_ok=True)

        if self.ui:
            self.monitor_dir = self.log_dir / "monitor"
            self.monitor_dir.mkdir(exist_ok=True)

        logger.info(f"{self.__class__.__name__} initialized. Logging to: {self.log_dir}")

    
    def _save_image(self, image: Image.Image) -> Path:
        """Save the image to the log directory."""
        if self.ui:
            self.ui.take_screenshot(self.monitor_dir, f"screenshot_{self.step_count}.jpg")
            
        image_path = self.image_dir / f"screenshot_{self.step_count}.png"
        image.save(image_path)
        return image_path
        
    def _encode_image(self, image: Image.Image) -> str:
        """Encode PIL Image to base64 string."""
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode()

    def store_observation(self, observation: Dict[str, Any], 
                          prev_action: Optional[str] = None) -> None:
        """Store the observation in the history."""
        image = observation['screen']
        buttons = observation['buttons']
        
        # Save image to log directory
        image_path = self._save_image(image)
        self.file_logger.info(f"Saved observation image to {image_path}")
        
        # Prepare the image for API
        base64_image = self._encode_image(image)
        
        # Prepare user message with game state
        if prev_action:
            user_content = [
                {
                    "type": "text",
                    "text": f"You previously pressed: {prev_action}"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{base64_image}"
                    }
                }
            ]
        else:
            user_content = [
                {"type": "text", "text": "Your current screen is:"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{base64_image}"
                    }
                }
            ]
        self.add_to_history("user", user_content, has_image=True)
        
    async def _prepare_messages(self) -> List[Dict[str, str]]:
        """Prepare the message list for LLM generation."""
        messages = [{"role": m.role, "content": m.content} for m in self.context_history]
        messages.append({
            "role": "user",
            "content": f"{REFLECTION_PROMPT}\n\n[Your current reflection memory]:\n{self.reflection_memory}"
        })
        messages.append({
            "role": "user",
            "content": f"{self.gba_prompt}"
        })
        return messages

    def _update_reflection_memory(self, response: str) -> None:
        """Extract and update reflection memory from LLM response."""
        reflection_match = re.search(r'```reflection\s*(.*?)\s*```', response, re.DOTALL)
        if reflection_match:
            self.reflection_memory = reflection_match.group(1).strip()
            self.save_reflection()

    def _add_response_to_history(self, response: str) -> None:
        """Add the LLM response to conversation history, excluding reflection content."""
        response_without_reflection = re.sub(r'```reflection\s*.*?\s*```', '', response, flags=re.DOTALL).strip()

        # Really annoying thing to patch, but some models will generate this tag.
        if not response_without_reflection.startswith("[Your thought]:"):
            response_without_reflection = "[Your thought]: " + response_without_reflection
        self.add_to_history("assistant", response_without_reflection)

    def _update_ui_state(self, actions: Optional[str] = None) -> None:
        """Update UI elements with current state."""
        if self.ui:
            self.ui.update_cost(self.llm_client.get_total_cost())
            self.ui.update_context_history(self.context_history)
            self.ui.update_steps_count(self.step_count)
            if actions is not None:
                self.ui.update_executing_action(actions)
                self.ui.update_last_action(self.action)

    async def get_action(self) -> Dict[str, bool] | List[Dict[str, bool]]:
        """
        Get the next action based on the current game observation.
        
        Returns:
            Dictionary mapping button names to boolean values (pressed/not pressed)
        """
        self.step_count += 1

        # Update UI state
        self._update_ui_state("")
        
        # Prepare and send messages to LLM
        messages = await self._prepare_messages()
        start_time = time.time()
        response = await self.llm_client.generate_response(
            system_message=self.system_prompt,
            messages=messages
        )
        response_time = time.time() - start_time

        # Handle error response
        if response.startswith("Error:"):
            return None

        # Log response time
        self.file_logger.info(f"Response time: {response_time:.2f}s")
        
        # Process reflection and update history
        self._update_reflection_memory(response)
        self._add_response_to_history(response)
        

        # Parse and return button states
        self.prev_action = self.action
        try:
            actions = parse_actions_response(response)
            button_states = convert_to_dict(actions)
            self.action = actions
            
            self._update_ui_state(actions)
            return [button_states] if not isinstance(button_states, list) else button_states
        except Exception as e:
            self.file_logger.error(f"Error parsing response: {e}")
            return None


class WebBrowsingVGAgent(VideoGameBenchAgent):
    """
    Agent that uses the ReACT method to browse the web and complete tasks.
    """
    def __init__(
        self,
        model: str,
        api_key: str,
        game: str,
        task_prompt: str,
        headless: bool = False,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        max_history_tokens: int = 4000,  # Add this parameter
        context_window: int = 10,        # Add this parameter
        lite: bool = False,
        press_key_delay: int = 100,
        log_dir: Optional[Path] = None,
        enable_ui: bool = False,
        api_base: Optional[str] = None
    ):
        """
        Initialize the web browsing agent.
        
        Args:
            model: The LLM model to use
            api_key: The API key for the model provider
            game: The name of the game to play
            headless: Whether to run the browser in headless mode
            temperature: The temperature for LLM generation
            max_tokens: The maximum number of tokens to generate
            max_history_tokens: The maximum number of tokens to keep in history
            context_window: Number of recent timesteps to include
            lite: Whether to run in lite mode with reduced functionality
            log_dir: Optional custom log directory path
            enable_ui: Whether to enable the UI monitor
        """
        super().__init__(
            model=model,
            api_key=api_key,
            game=game,
            headless=headless,
            temperature=temperature,
            max_tokens=max_tokens,
            max_history_tokens=max_history_tokens,
            context_window=context_window,
            log_dir=log_dir,
            enable_ui=enable_ui,
            task_prompt=task_prompt,
            api_base=api_base
        )
        
        # Initialize browser controller
        self.frames: List[bytes] = []
        self.task: Optional[str] = None

        # Game-specific settings
        self.press_key_delay = press_key_delay
        
        # Additional WebBrowsingVGAgent-specific log file
        self.conversation_log_file = self.log_dir / "conversation_history.json"
        self.conversation_log_file.touch()

        self.system_instruction_prompt = SYSTEM_PROMPTS["dos"]
        self.system_prompt = {
            "role": "system", 
            "content": f"{self.system_instruction_prompt}\n\n{task_prompt}"
        }
        self.lite = lite

        self.screenshot_dir = self.log_dir / "game_screen"
        self.monitor_dir = self.log_dir / "monitor"
        self.screenshot_dir.mkdir(exist_ok=True)
        self.monitor_dir.mkdir(exist_ok=True)

    async def start(self) -> None:
        """
        Start the agent with any initial information.
        """
        pass

    async def stop(self) -> None:
        """
        DOS Agent deconstructor.
        """
        pass
        
    
    async def save_to_history(self, frame):
        """
        Save the frame to the LLM context history by first decoding
        to the right format.
        """
        base64_image = base64.b64encode(frame).decode("utf-8")
        user_content = [
            {
                "type": "text",
                "text": "Frame:"
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                }
            }
        ]
        self.add_to_history("user", user_content, has_image=True)

    
    async def store_observation(self, frames: List[bytes]) -> None:
        # Save screenshot as current "step" representative frame
        for i, frame in enumerate(frames):
            self.frames.append(frame)
            await self.save_to_history(frame)

            filename = f"game_screen_step_{self.step_count}.jpg" if len(frames) == 1 else f"game_screen_step_{self.step_count}_{i}.jpg"
            screenshot_path = self.screenshot_dir / filename

            # Save the screenshot to the log directory
            with open(screenshot_path, "wb") as f:
                f.write(frame)
            if self.ui:
                self.ui.take_screenshot(self.monitor_dir, filename)
            self.file_logger.info(f"Saved step {self.step_count} screenshot{' ' + str(i) if i > 0 else ''} to {screenshot_path}")
    
    async def pre_action(self, 
                         action: str,
                         action_input: str,
                         lite: bool = False) -> None:
        """
        Pre-action hook for the agent.
        """
        # Update UI with currently executing action
        if self.ui:
            self.ui.update_executing_action(f"{action}: {action_input}")

        if self.file_logger is not None:
            self.file_logger.info(f"Executing action: {action} with input: {action_input}")
            if lite: self.file_logger.info("Lite mode is enabled, pausing game with Alt+Pause key...")
        
        self.frames = []
    
    async def post_action(self, 
                          info: str,
                          frames: List[bytes],
                          action: str,
                          action_input: str) -> None:
        """
        Post-action hook for the agent.
        """
        # Update conversation history with observation

        self.context_history.append(Message(role="user", content=f"Observation: {info}"))

        await self.store_observation(frames)

        # Save conversation history to consolidated file
        messages = [{"role": m.role, "content": m.content} for m in self.context_history 
                   if not (isinstance(m.content, list) and any(item.get("type") == "image_url" for item in m.content))]
        with open(self.conversation_log_file, "a", encoding="utf-8") as f:
            json.dump(messages[-2:], f, indent=2, ensure_ascii=False)  # Only save the latest exchange
        self.file_logger.info(f"Saved conversation history to {self.conversation_log_file}")

        # Logging logic, verbose
        self.file_logger.info(f"Observation: {info}")
        logger.info(f"Observation: {info}")
        logger.info(f"Current memory: {self.reflection_memory}")

        # Update UI if needed
        if self.ui:
            self.ui.update_context_history(messages)
            self.ui.update_executing_action("")
            if action and action_input:
                self.ui.update_last_action(action + ", " + action_input)
            self.ui.update_steps_count(self.step_count)
    
    async def get_action(self, 
                        task: str,
                        browser: BrowserController,
                        step: int) -> Dict[str, bool]:
        self.task = self.task_prompt + "\n\n" + task 

        # Generate the next action using ReACT
        start_time = time.time()
        # Add reflection prompt to the task
        task_with_reflection = (
            f"{REFLECTION_PROMPT}\n\n[memory]:\n{self.reflection_memory}"
            f"Your mouse is currently at coordinates: {browser.current_mouse_position}. Move it with move or drag actions."
            f"{self.task_prompt}\n\n"
        )
        
        # Prune screenshots to only keep the latest context_window
        if len(self.frames) > self.context_window:
            self.frames = self.frames[-self.context_window:]


        print("Frames going in", len(self.frames))
        messages = [{"role": m.role, "content": m.content} for m in self.context_history]
        react_response = await self.llm_client.generate_react_response(
            task=task_with_reflection,
            system_message=self.system_prompt,
            history=messages,
            screenshots=self.frames
        )
        if react_response is None:
            return None, None
        else:
            response_time = time.time() - start_time
            
            # Update the cost display in UI
            if self.ui:
                self.ui.update_cost(self.llm_client.get_total_cost())
            
            # Update reflection memory from the JSON response
            memory = react_response.get("memory", "")
            if memory:  # Only update if memory exists and is not empty
                self.reflection_memory = memory
            
            # Save reflection to consolidated file
            if self.reflection_memory:
                self.save_reflection()
            
            # Log the thought process
            thought = react_response.get("thought", "")
            action = react_response.get("action", "")
            action_input = react_response.get("action_input", "")
            
            self.file_logger.info(f"Response time: {response_time:.2f}s")
            self.file_logger.info(f"Thought: {thought}")
            self.file_logger.info(f"Action: {action}, Input: {action_input}")
            logger.info(f"Thought: {thought}")
            logger.info(f"Action: {action}, Input: {action_input}")

            # Update the conversation history
            self.context_history.append(Message(role="assistant", content=json.dumps({
                    "thought": thought,
                    "action": action,
                    "action_input": action_input
                })
            ))

            self.step_count = step + 1
            return action, action_input

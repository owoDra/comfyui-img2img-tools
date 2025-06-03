"""
title: ComfyUI Img2Img Tool
Author: owoDra
author_url: https://github.com/owoDra
git_url: https://github.com/owoDra/comfyui-img2img-tools.git
description: Tool that integrates Comfyui and Open Web UI to generate images based on reference images presented during a conversation
required_open_webui_version: 0.6.0
requirements: requests
version: 1.0.0
license: MIT
"""

import requests
import json
import asyncio
import re
import random
from typing import List, Dict, Optional, Any, Tuple, Callable
from pydantic import BaseModel, Field


""" ===========================
Constants
=========================== """

EXAMPLE_WORKFLOW_JSON = """
{
  "1": {
    "inputs": {
      "ckpt_name": ""
    },
    "class_type": "CheckpointLoaderSimple",
    "_meta": {
      "title": "Load Checkpoint"
    }
  },
  "2": {
    "inputs": {
      "pixels": [
        "6",
        0
      ],
      "vae": [
        "1",
        2
      ]
    },
    "class_type": "VAEEncode",
    "_meta": {
      "title": "VAE Encode"
    }
  },
  "3": {
    "inputs": {
      "filename_prefix": "ComfyUI",
      "images": [
        "4",
        0
      ]
    },
    "class_type": "SaveImage",
    "_meta": {
      "title": "Save Image"
    }
  },
  "4": {
    "inputs": {
      "samples": [
        "5",
        0
      ],
      "vae": [
        "1",
        2
      ]
    },
    "class_type": "VAEDecode",
    "_meta": {
      "title": "VAE Decode"
    }
  },
  "5": {
    "inputs": {
      "seed": 0,
      "steps": 20,
      "cfg": 8,
      "sampler_name": "euler",
      "scheduler": "normal",
      "denoise": 1,
      "model": [
        "1",
        0
      ],
      "positive": [
        "8",
        0
      ],
      "negative": [
        "7",
        0
      ],
      "latent_image": [
        "9",
        0
      ]
    },
    "class_type": "KSampler",
    "_meta": {
      "title": "KSampler (inspire)"
    }
  },
  "6": {
    "inputs": {
      "data": ""
    },
    "class_type": "LoadImageFromBase64",
    "_meta": {
      "title": "Load Image (Base64)"
    }
  },
  "7": {
    "inputs": {
      "text": "negative",
      "clip": [
        "1",
        1
      ]
    },
    "class_type": "CLIPTextEncode",
    "_meta": {
      "title": "CLIP Text Encode (Prompt)"
    }
  },
  "8": {
    "inputs": {
      "text": "positive",
      "clip": [
        "1",
        1
      ]
    },
    "class_type": "CLIPTextEncode",
    "_meta": {
      "title": "CLIP Text Encode (Prompt)"
    }
  },
  "9": {
    "inputs": {
      "upscale_method": "nearest-exact",
      "width": 512,
      "height": 512,
      "crop": "disabled",
      "samples": [
        "2",
        0
      ]
    },
    "class_type": "LatentUpscale",
    "_meta": {
      "title": "Latent Expand"
    }
  }
}
"""
EXAMPLE_NODE_DEFINE_JSON = """
{
    "Model": {
        "param": "ckpt_name",
        "id": "1",
        "default": ""
    },
    "Image": {
        "param": "data",
        "id": "6",
        "default": ""
    },
    "Width": {
        "param": "width",
        "id": "9",
        "default": 512
    },
    "Height": {
        "param": "height",
        "id": "9",
        "default": 512
    },
    "PositivePrompt": {
        "param": "text",
        "id": "8",
        "default": "masterpiece,best quality,amazing quality"
    },
    "NegativePrompt": {
        "param": "text",
        "id": "7",
        "default": "bad quality,worst quality,worst detail,sketch,censor"
    },
    "Seed": {
        "param": "seed",
        "id": "5",
        "default": -1
    },
    "Steps": {
        "param": "steps",
        "id": "5",
        "default": 15
    },
    "Denoise": {
        "param": "denoise",
        "id": "5",
        "default": 0.7
    }
}
"""
EXAMPLE_API_BASEURL = "http://localhost:8188"


""" ===========================
Exception
=========================== """

class ToolError(Exception):
    def __init__(self, message: str):
        super().__init__(message)
    
    def __str__(self) -> str:
        return f"Error: {super().__str__()}"


""" ===========================
Helpers
=========================== """

class EventEmitter:
    def __init__(self, event_emitter: Callable[[dict], Any]):
        self.event_emitter = event_emitter

    async def emit(self, description="Unknown state", done=False, hidden=False):
        if self.event_emitter:
            await self.event_emitter(
                {
                    "type": "status",
                    "data": {
                        "description": description,
                        "done": done,
                        "hidden": hidden
                    },
                }
            )

    async def emit_message(self, content="msg"):
        if self.event_emitter:
            await self.event_emitter(
                {
                    "type": "chat:message:delta",
                    "data": { "content": content },
                }
            )

class WorkflowProcesor:
    def __init__(
        self,
        api_baseurl: str,
        workflow_json_str: str,
        node_define_json_str: str,
        req_timeout: int,
        req_interval: int,
    ):
        self.workflow_endpoint  = f"{api_baseurl}/prompt"
        self.history_endpoint   = f"{api_baseurl}/history"
        self.image_endpoint     = f"{api_baseurl}/view"
        self.workflow           = json.loads(workflow_json_str)
        self.node_define        = json.loads(node_define_json_str)
        self.req_timeout        = req_timeout
        self.req_interval       = req_interval

        self.result = ""

    def _1_build_workflow(self, prompt: str, image_base64: str):
        for label, node_def in self.node_define.items():
            try:
                node_id    = node_def["id"]
                node_param = node_def["param"]
                node_value = node_def["default"]

                if label == "PositivePrompt":
                    node_value += f",{prompt}"

                elif label == "Image":
                    node_value = image_base64

                elif label == "Seed":
                    if node_value is -1:
                        node_value = random.randint(0, 2**32 - 1)

                self.workflow[node_id]["inputs"][node_param] = node_value

                if label != "Image":
                    self.result += f"{label}: {node_value}, "

            except KeyError as e:
                raise ToolError(f"Invalid NodeDef or Workflow | {e}")

    async def _2_request_workflow(self) -> str:
        try:
            response = await asyncio.get_running_loop().run_in_executor(
                None,
                lambda: requests.post(
                    self.workflow_endpoint,
                    json={ "prompt": self.workflow },
                    timeout=self.req_timeout + 10,
                ))

            response.raise_for_status()

            return response.json()["prompt_id"]

        except requests.exceptions.RequestException as e:
            raise ToolError(f"Workflow Request Failed | {e} | {self.workflow}")

    async def _3_wait_result(self, prompt_id: str) -> str:
        history_endpoint = f"{self.history_endpoint}/{prompt_id}"
        max_retries = (self.req_timeout // self.req_interval if self.req_interval > 0 else 1)

        for retries in range(max_retries):
            await asyncio.sleep(self.req_interval)
            try:
                history_response = await asyncio.get_running_loop().run_in_executor(
                    None,
                    lambda: requests.get(
                        history_endpoint, timeout=self.req_timeout
                    ))

                if history_response.status_code == 200:
                    history_data = history_response.json()
                    if prompt_id in history_data and history_data[prompt_id].get("outputs"):
                        outputs = history_data[prompt_id]["outputs"]
                        for _, node_out in outputs.items():
                            if "images" in node_out:
                                for img_data in node_out["images"]:
                                    if img_data.get("type") == "output":
                                        filename = img_data["filename"]
                                        subfolder = img_data.get("subfolder", "")
                                        img_url = f"{self.image_endpoint}?filename={requests.utils.quote(filename)}&subfolder={requests.utils.quote(subfolder)}&type=output"
                                        return f"![]({img_url})"

            except requests.exceptions.Timeout:
                pass
            except requests.exceptions.RequestException as e:
                raise ToolError(f"Error: Image Request Failed: {e}")
            
        raise ToolError("Image Request Timeout")

    async def execute(
        self,
        prompt: str,
        image_base64: str,
    ) -> str:
        self._1_build_workflow(prompt, image_base64)
        prompt_id = await self._2_request_workflow()
        image_url = await self._3_wait_result(prompt_id)
        return f"{self.result}Image: {image_url}" 
        
class Base64ImageProcesor:
    def __init__(self):
        pass

    def _extract_potential_image_source(
        self,
        msg: Dict[str, Any],
    ) -> Optional[str]:
        content = msg.get("content")

        if isinstance(content, list):
            for item in content:
                if item.get("type") == "image_url":
                    img_url_data = item.get("image_url")

                    if isinstance(img_url_data, dict):
                        return img_url_data.get("url")
                    elif isinstance(img_url_data, str):
                        return img_url_data
        return None

    def _1_extract_all_potential_image_sources(
        self, messages_history: List[Dict[str, Any]]
    ) -> List[str]:
        sources: List[str] = []

        for msg in messages_history:
            url = self._extract_potential_image_source(msg)
            if url:
                sources.append(url)

        return sources

    def _2_resolve_selector(self, selector: Any) -> int:
        NOT_FOUND = -2

        if isinstance(selector, str):
            match = re.search(r"\d+", selector)
            if match:
                return int(match.group(0))

        elif isinstance(selector, int):
            return selector

        elif isinstance(selector, float):
            return int(selector)

        elif isinstance(selector, List):
            for item in selector:
                value = self._2_resolve_selector(item)
                if value is not NOT_FOUND:
                    return value

        return NOT_FOUND

    def _3_select_source(self, all_sources: List[str], selector: int) -> Optional[str]:
        if 0 <= selector < len(all_sources):
            return all_sources[selector]

        try:
            return all_sources[-1]

        except IndexError:
            return None

    def _4_get_base64_data_source(
        self,
        source: Optional[str],
    ) -> Optional[str]:
        if not source:
            return None

        if source.startswith("data:") and ";base64," in source:
            try:
                return source.split(",", 1)[1]
            except IndexError:
                return None

        return None

    def execute(
        self,
        messages_history: List[Dict[str, Any]],
        selector: Any,
    ) -> str:
        all_sources = self._1_extract_all_potential_image_sources(messages_history)
        resolved_selector = self._2_resolve_selector(selector)
        source = self._3_select_source(all_sources, resolved_selector)
        base64_data = self._4_get_base64_data_source(source)

        if base64_data:
            return base64_data

        raise ToolError("Could not get image as base64 data")


""" ===========================
Tool
=========================== """

class Tools:
    class Valves(BaseModel):
        # Comfy Settings
        COMFYUI_API_BASEURL: str = Field(
            default=EXAMPLE_API_BASEURL, description=EXAMPLE_API_BASEURL
        )
        COMFYUI_IMG2IMG_WORKFLOW_JSON: str = Field(
            default=EXAMPLE_WORKFLOW_JSON, description="Enter workflow json here"
        )
        COMFYUI_IMG2IMG_NODE_DEFINE_JSON: str = Field(
            default=EXAMPLE_NODE_DEFINE_JSON,
            description="Allowed labels: Model, Image, Width, Height, PositivePrompt, NegativePrompt, Seed, Steps",
        )

        # Request Settings
        REQUEST_TIMEOUT_SECONDS: int = Field(
            default=120, description="Request Timeout (sec)"
        )
        REQUEST_INTERVAL_SECONDS: int = Field(
            default=5, description="Request Interval (sec)"
        )

    def __init__(self):
        self.valves = self.Valves()

    async def img2img(
        self,
        prompt: str,
        image_selector: Any,
        __messages__: List[Dict[str, Any]],
        __event_emitter__: Callable[[dict], Any],
    ) -> str:
        """
        Generate an image from a given prompt and a number of attached images

        :param prompt: Prompts described in English used for image generation
        :param image_selector: Index for selecting an image source from history.
        :return Generated image and parameters used to generate it. The image must be included in the response.
        """

        event_emitter = EventEmitter(__event_emitter__)

        try:
            # 1. Retrieve image data as base64
            base64_image_procesor = Base64ImageProcesor()
            base64_data = base64_image_procesor.execute(__messages__, image_selector)

            # 2. Execute workflow
            workflow_procesor = WorkflowProcesor(
                self.valves.COMFYUI_API_BASEURL,
                self.valves.COMFYUI_IMG2IMG_WORKFLOW_JSON,
                self.valves.COMFYUI_IMG2IMG_NODE_DEFINE_JSON,
                self.valves.REQUEST_TIMEOUT_SECONDS,
                self.valves.REQUEST_INTERVAL_SECONDS,
            )
            return await workflow_procesor.execute(prompt, base64_data)

        except ToolError as e:
            return f"{e}"
        except Exception as e:
            return f"Error: Unknown | {e}"

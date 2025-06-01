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

import base64
import requests
import json
import asyncio
import re
import random
from typing import List, Dict, Optional, Any, Tuple, Callable, Awaitable
from pydantic import BaseModel, Field

""" ===========================
Constants
=========================== """

LABEL_MODEL = "Model"
LABEL_IMAGE = "Image"
LABEL_WIDTH = "Width"
LABEL_HEIGHT = "Height"
LABEL_POSITIVE_PROMPT = "PositivePrompt"
LABEL_NEGATIVE_PROMPT = "NegativePrompt"
LABEL_SEED = "Seed"
LABEL_STEPS = "Steps"

EXAMPLE_WORKFLOW_JSON = """
{
  "1": {
    "inputs": {
      "ckpt_name": "waiNSFWIllustrious_v140.safetensors"
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
    "{LABEL_MODEL}": {
        "param": "ckpt_name",
        "id": "1",
        "default": "waiNSFWIllustrious_v140.safetensors"
    },
    "{LABEL_IMAGE}": {
        "param": "data",
        "id": "6",
        "default": ""
    },
    "{LABEL_WIDTH}": {
        "param": "width",
        "id": "9",
        "default": "512"
    },
    "{LABEL_HEIGHT}": {
        "param": "height",
        "id": "9",
        "default": "512"
    },
    "{LABEL_POSITIVE_PROMPT}": {
        "param": "text",
        "id": "7",
        "default": "masterpiece,best quality,amazing quality"
    },
    "{LABEL_NEGATIVE_PROMPT}": {
        "param": "text",
        "id": "8",
        "default": "bad quality,worst quality,worst detail,sketch,censor"
    },
    "{LABEL_SEED}": {
        "param": "seed",
        "id": "5",
        "default": "-1"
    },
    "{LABEL_STEPS}": {
        "param": "steps",
        "id": "5",
        "default": "15"
    }
}
"""
EXAMPLE_API_BASEURL = "http://localhost:8188"


""" ===========================
Helpers
=========================== """


class EventEmitter:
    def __init__(self, event_emitter: Callable[[dict], Any] = None):
        self.event_emitter = event_emitter

    async def emit(self, status="in_progress", description="Unknown state", done=False):
        if self.event_emitter:
            await self.event_emitter(
                {
                    "type": "status",
                    "data": {
                        "status": status,
                        "description": description,
                        "done": done,
                    },
                }
            )

    async def emit_message(self, content="msg"):
        if self.event_emitter:
            await self.event_emitter(
                {
                    "type": "message",
                    "data": {
                        "content": content,
                        "type": "text",
                        "role": "assistant",
                    },
                }
            )


class WorkflowHelper:
    def __init__(
        self,
        api_baseurl: str,
        workflow_json_str: str,
        node_define_json_str: str,
        req_timeout: int,
        req_interval: int,
    ):
        self.workflow_endpoint = f"{api_baseurl}/prompt"
        self.history_endpoint = f"{api_baseurl}/history"
        self.image_endpoint = f"{api_baseurl}/view"
        self.default_workflow = json.loads(workflow_json_str)
        self.node_define = json.loads(node_define_json_str)
        self.req_timeout = req_timeout
        self.req_interval = req_interval

        for label, node_def in self.node_define.items():
            try:
                node_id = node_def["id"]
                node_param = node_def["param"]
                node_default = node_def["default"]

                self.default_workflow[node_id]["inputs"][node_param] = node_default

            except KeyError:
                return f"Error: Invalid NodeDef or Workflow"

    async def execute(
        self,
        prompt: str,
        image_base64: str,
        positive: Optional[str] = None,
        negative: Optional[str] = None,
        model: Optional[str] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        seed: Optional[int] = None,
        steps: Optional[int] = None,
    ):
        workflow = self.default_workflow

        # build workflow
        try:
            node_id = self.node_define.items[LABEL_POSITIVE_PROMPT]["id"]
            node_param = self.node_define.items[LABEL_POSITIVE_PROMPT]["param"]
            workflow[node_id]["inputs"][node_param] += f",{prompt}"

            if positive:
                workflow[node_id]["inputs"][node_param] += f",{positive}"

            node_id = self.node_define.items[LABEL_IMAGE]["id"]
            node_param = self.node_define.items[LABEL_IMAGE]["param"]
            workflow[node_id]["inputs"][node_param] += image_base64

            if negative:
                node_id = self.node_define.items[LABEL_NEGATIVE_PROMPT]["id"]
                node_param = self.node_define.items[LABEL_NEGATIVE_PROMPT]["param"]
                workflow[node_id]["inputs"][node_param] += f",{negative}"

            if model:
                node_id = self.node_define.items[LABEL_MODEL]["id"]
                node_param = self.node_define.items[LABEL_MODEL]["param"]
                workflow[node_id]["inputs"][node_param] = model

            if width:
                node_id = self.node_define.items[LABEL_WIDTH]["id"]
                node_param = self.node_define.items[LABEL_WIDTH]["param"]
                workflow[node_id]["inputs"][node_param] = width

            if height:
                node_id = self.node_define.items[LABEL_HEIGHT]["id"]
                node_param = self.node_define.items[LABEL_HEIGHT]["param"]
                workflow[node_id]["inputs"][node_param] = height

            if seed:
                node_id = self.node_define.items[LABEL_SEED]["id"]
                node_param = self.node_define.items[LABEL_SEED]["param"]
                workflow[node_id]["inputs"][node_param] = seed

            if workflow[node_id]["inputs"][node_param] == -1:
                workflow[node_id]["inputs"][node_param] = random.randint(0, 2**32 - 1)

            if steps:
                node_id = self.node_define.items[LABEL_STEPS]["id"]
                node_param = self.node_define.items[LABEL_STEPS]["param"]
                workflow[node_id]["inputs"][node_param] = steps

        except KeyError:
            return f"Error: Invalid execution parameters"

        # request workflow
        try:
            response = await loop.run_in_executor(
                None,
                lambda: requests.post(
                    self.workflow_endpoint, json=workflow, timeout=self.req_timeout + 10
                ),
            )
            response.raise_for_status()
            prompt_response = response.json()

            if "prompt_id" not in prompt_response:
                return f"Error: Response without prompt id"

            prompt_id = prompt_response["prompt_id"]

        except requests.exceptions.RequestException as e:
            return f"Error: Workflow Request Failed: {e}"

        # request image
        history_endpoint = f"{self.history_endpoint}/{prompt_id}"
        max_retries = (
            self.req_timeout // self.req_interval if self.req_interval > 0 else 1
        )

        for retries in range(max_retries):
            await asyncio.sleep(self.req_interval)
            try:
                history_response = await loop.run_in_executor(
                    None,
                    lambda: requests.get(history_endpoint, timeout=self.req_timeout),
                )

                if hist_response.status_code == 200:
                    history_data = hist_response.json()
                    if prompt_id in history_data and history_data[prompt_id].get(
                        "outputs"
                    ):
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
                if retries == max_retries - 1:
                    return f"Error: Image Request Timeout"
            except requests.exceptions.RequestException as e:
                return f"Error: Image Request Failed: {e}"


class ImageHelper:
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

    def _extract_all_potential_image_sources(
        self, messages_history: List[Dict[str, Any]]
    ) -> List[str]:
        sources: List[str] = []

        for msg in messages_history:
            url = self._extract_potential_image_source
            if url:
                sources.append(url)

        return sources

    def _select_source(
        self, all_sources: List[str], selector: Optional[str], regex: Optional[str]
    ) -> Tuple[Optional[str], Optional[str]]:
        if not all_sources:
            return None, "Error: No sources available"

        if selector:
            if regex:
                match = re.search(regex, selector)

                if match:
                    number_str = match.group(1)
                    number = int(number_str)

                    if 0 <= number < len(all_sources):
                        return all_sources[number], None

            else:
                return all_sources[int(selector)], None

        return all_sources[-1], None

    def _get_base64_data_from_data_uri(
        self,
        source: str,
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        :param source: The data URI string (e.g., "data:image/png;base64,...").
        :return: A tuple containing:
                    - The Base64 encoded data string (data part only), or None on failure.
                    - The original full data URI for preview, or None on failure.
                    - An error message string if an error occurred, otherwise None.
        """

        if not source.startswith("data:") or ";base64," not in source:
            return None, "Error: URI is not Base64 data"

        try:
            data_part = source.split(",", 1)[1]
            return data_part, None
        except IndexError:
            err_msg = "Error: No data present in URI"
        except Exception as e:
            err_msg = f"Error: {e}"
        return None, err_msg

    def get_image_as_base64_data(
        self,
        messages_history: List[Dict[str, Any]],
        selector: Optional[str],
        regex: Optional[str],
    ) -> str:
        all_sources = self._extract_all_potential_image_sources(messages_history)
        source, err = self._select_source(all_sources, selector, regex)

        if err:
            return err

        if source.startswith("data:"):
            data, err = self._get_base64_data_from_data_uri(source)
            return data if data else err

        return "Error: Could not get base64 data"


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
            default=EXAMPLE_WORKFLOW_JSON, description=EXAMPLE_WORKFLOW_JSON
        )
        COMFYUI_IMG2IMG_NODE_DEFINE_JSON: str = Field(
            default=EXAMPLE_NODE_DEFINE_JSON, description=EXAMPLE_NODE_DEFINE_JSON
        )

        # Request Settings
        REQUEST_TIMEOUT_SECONDS: int = Field(
            default=120, description="Request Timeout (sec)"
        )
        REQUEST_INTERVAL_SECONDS: int = Field(
            default=5, description="Request Interval (sec)"
        )

        # Prompt Settings
        IMAGE_SELECTOR_PATTERN: str = Field(
            default=r"\[img-(\d+)\]",
            description="Regex to get the value from the image number passed from LLM",
        )

    def __init__(self):
        self.valves = self.Valves()

        self.workflow = WorkflowHelper(
            self.valves.COMFYUI_API_BASEURL,
            self.valves.COMFYUI_IMG2IMG_WORKFLOW_JSON,
            self.valves.COMFYUI_IMG2IMG_NODE_DEFINE_JSON,
            self.valves.REQUEST_TIMEOUT_SECONDS,
            self.valves.REQUEST_INTERVAL_SECONDS,
        )

    async def img2img(
        self,
        prompt: str,
        image_selector: Optional[str] = None,
        positive: Optional[str] = None,
        negative: Optional[str] = None,
        model: Optional[str] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        seed: Optional[int] = None,
        steps: Optional[int] = None,
        __messages__: List[Dict[str, Any]] = None,
        __event_emitter__: Callable[[dict], Any] = None,
    ) -> str:
        """
        Generate an image from a given prompt and a number of attached images

        :param prompt: prompt to use for image generation
        :param image_selector: User's preference for selecting an image source from history.
        :param positive: Additional positive prompts used for image generation.
        :param negative: Additional negative prompts used for image generation.
        :param model: Name of the generative model used to generate the image.
        :param width: Width of the generated image.
        :param height: Height of the generated image.
        :param seed: Seed value used for image generation.
        :param steps: Number of steps used to generate images.
        """

        event_emitter = EventEmitter(__event_emitter__)

        data = ImageHelper().get_image_as_base64_data(
            __messages__, image_selector, self.valves.IMAGE_SELECTOR_PATTERN
        )

        if data.startswith("Error:"):
            return data

        result = self.workflow.execute(
            prompt,
            data,
            positive,
            negative,
            model,
            width,
            height,
            seed,
            steps,
        )

        if result.startswith("Error:"):
            return result

        return result

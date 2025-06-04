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

EXAMPLE_WORKFLOW_JSON = """{
  "1": {
    "inputs": {
      "ckpt_name": "model.safetensors"
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
EXAMPLE_NODE_DEFINE_JSON = """{
    "Model": {
        "param": "ckpt_name",
        "id": "1"
    },
    "Image": {
        "param": "data",
        "id": "6"
    },
    "Width": {
        "param": "width",
        "id": "9"
    },
    "Height": {
        "param": "height",
        "id": "9"
    },
    "PositivePrompt": {
        "param": "text",
        "id": "8"
    },
    "NegativePrompt": {
        "param": "text",
        "id": "7"
    },
    "Seed": {
        "param": "seed",
        "id": "5"
    },
    "Steps": {
        "param": "steps",
        "id": "5"
    },
    "Denoise": {
        "param": "denoise",
        "id": "5"
    }
}
"""
EXAMPLE_API_BASEURL = "http://localhost:8188"

EXAMPLE_WIDTH  = 512
EXAMPLE_HEIGHT = 512
EXAMPLE_SEED   = -1
EXAMPLE_STEPS  = 15

EXAMPLE_DENOISE_OVERRIDE       = 0.0
EXAMPLE_POSITIVE_SYSTEM_PROMPT = "masterpiece,best quality,amazing quality"
EXAMPLE_NEGATIVE_SYSTEM_PROMPT = "bad quality,worst quality,worst detail,sketch,censor"

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

class ParameterResolver:
    def __init__(self):
        pass

    def resolve_image_selector(self, selector: Any) -> int:
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
                value = self.resolve_image_selector(item)
                if value is not NOT_FOUND:
                    return value

        return NOT_FOUND

    def resolve_model(self, model: str, model_override: str) -> str:
        return model_override if model_override != "" else model

    def resolve_denoise(self, denoise: float, denoise_override: float) -> float:
        final_value = denoise_override if denoise_override != 0.0 else denoise
        if final_value < 0.0:
            return 0.0
        elif final_value > 1.0:
            return 1.0
        return final_value
    
    def resolve_prompt(self, system_prompt: str, prompt: str) -> str:
        return f"{system_prompt},{prompt}"
    
    def resolve_seed(self, seed: int) -> int:
        return random.randint(0, 2**32 - 1) if seed == -1 else seed

class Base64ImageResolver:
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
        self, messages_history: Optional[List[Dict[str, Any]]]
    ) -> List[str]:
        sources: List[str] = []

        if messages_history:
            for msg in messages_history:
                url = self._extract_potential_image_source(msg)
                if url:
                    sources.append(url)

        return sources

    def _2_select_source(self, all_sources: List[str], selector: int) -> Optional[str]:
        if 0 <= selector < len(all_sources):
            return all_sources[selector]

        try:
            return all_sources[-1]

        except IndexError:
            return None

    def _3_get_base64_data_source(
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

    def resolve(
        self,
        messages_history: Optional[List[Dict[str, Any]]],
        resolved_selector: int,
    ) -> str:
        all_sources = self._1_extract_all_potential_image_sources(messages_history)
        source      = self._2_select_source(all_sources, resolved_selector)
        base64_data = self._3_get_base64_data_source(source)

        if base64_data:
            return base64_data

        raise ToolError("Could not get image as base64 data")

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

    def _build_param(self, label: str, value: Any):
        try:
            node_def   = self.node_define[label]
            node_id    = node_def["id"]
            node_param = node_def["param"]

            self.workflow[node_id]["inputs"][node_param] = value

        except KeyError as e:
            raise ToolError(f"Invalid NodeDef or Workflow | {e}")

    def build(
        self, 
        image_base64: str,
        positive_prompt: str,
        negative_prompt: str,
        denoise: float,
        model: str, 
        seed: int,
        steps: int,
        width: int,
        height: int,
    ):
        self._build_param("Model", model)
        self._build_param("Image", image_base64)
        self._build_param("Width", width)
        self._build_param("Height", height)
        self._build_param("PositivePrompt", positive_prompt)
        self._build_param("NegativePrompt", negative_prompt)
        self._build_param("Seed", seed)
        self._build_param("Steps", steps)
        self._build_param("Denoise", denoise)
        
    async def _1_request_workflow(self) -> str:
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

    async def _2_wait_result(self, prompt_id: str) -> str:
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
                                        return img_url

            except requests.exceptions.Timeout:
                pass
            except requests.exceptions.RequestException as e:
                raise ToolError(f"Error: Image Request Failed: {e}")
            
        raise ToolError("Image Request Timeout")

    async def execute(self) -> str:
        prompt_id = await self._1_request_workflow()
        image_url = await self._2_wait_result(prompt_id)
        return f"![]({image_url})"
        
class ResultFormatter:
    def __init__(
        self,
        image_url: str,
        positive_promt: str,
        negative_promt: str,
        denoise: float,
    ):
        self.result = f"""
Image URL: {image_url}
Positive Prompt: {positive_promt}
Negative Prompt: {negative_promt}
Denoise: {denoise}
"""

""" ===========================
Tool
=========================== """

class Tools:
    class Valves(BaseModel):
        # Comfy Settings
        COMFYUI_API_BASEURL: str = Field(
            default=EXAMPLE_API_BASEURL, 
            description=EXAMPLE_API_BASEURL
        )
        COMFYUI_IMG2IMG_WORKFLOW_JSON: str = Field(
            default=EXAMPLE_WORKFLOW_JSON, 
            description="Enter workflow json here"
        )
        COMFYUI_IMG2IMG_NODE_DEFINE_JSON: str = Field(
            default=EXAMPLE_NODE_DEFINE_JSON,
            description="Allowed labels: Model, Image, Width, Height, PositivePrompt, NegativePrompt, Seed, Steps, Denoise",
        )
        MODEL: str = Field(
            default="", 
            description="Model file name used for image generation"
        )

        # Request Settings
        REQUEST_TIMEOUT_SECONDS: int = Field(
            default=120, description="Request Timeout (sec)"
        )
        REQUEST_INTERVAL_SECONDS: int = Field(
            default=5, description="Request Interval (sec)"
        )

        pass

    class UserValves(BaseModel):
        IMAGE_WIDTH: int = Field(
            default=EXAMPLE_WIDTH, 
            description="Width of the generated image"
        )
        IMAGE_HEIGHT: int = Field(
            default=EXAMPLE_HEIGHT, 
            description="Height of the generated image"
        )
        SEED: int = Field(
            default=EXAMPLE_SEED, 
            description="Seed for image generation"
        )
        STEPS: int = Field(
            default=EXAMPLE_STEPS, 
            description="Steps for image generation"
        )
        DENOISE_OVERRIDE: float = Field(
            default=EXAMPLE_DENOISE_OVERRIDE, 
            description="Override denoise of images to be generated. (Override disabled at 0.0)"
        )
        MODEL_OVERRIDE: str = Field(
            default="", 
            description="Override model file name used for image generation"
        )
        POSITIVE_SYSTEM_PROMPT: str = Field(
            default=EXAMPLE_POSITIVE_SYSTEM_PROMPT, 
            description="Positive system prompts used for image generation"
        )
        NEGATIVE_SYSTEM_PROMPT: str = Field(
            default=EXAMPLE_NEGATIVE_SYSTEM_PROMPT, 
            description="Negative system prompts used for image generation"
        )
        
        pass

    def __init__(self):
        self.valves = self.Valves()

    async def img2img(
        self,
        image_selector: Any,
        positive_prompt: str,
        negative_prompt: str,
        denoise: float,
        __user__: dict,
        __messages__: Optional[List[Dict[str, Any]]] = None,
        __event_emitter__: Optional[Callable[[dict], Any]] = None,
    ) -> str:
        """
        Generates an image from the specified English prompt and the attached image.

        :param image_selector: Index for selecting an image source from history.
                               0 is the first image attached, 1 is the second image attached.
                               If not specified, the most recently attached image will be used.

        :param positive_prompt: Positive prompt described in english, used to generate the image, which describes desired elements in the image.
                                Instructions that are more important can be coordinated by enclosing them in parentheses ().
                                Instructions enumerated in the prompt must be in English.

        :param negative_prompt: Negative prompts described in english, used to generate the image, which describes undesired elements in the image.
                                Instructions that are more important can be coordinated by enclosing them in parentheses ().
                                Instructions enumerated in the prompt must be in English.
                                
        :param denoise: A value indicating how much the generated image differs from the original image. 
                        The range is from 0.0 to 1.0. It gets more different each closer to 1.0. 
                        0.0 is exactly same as the original image, and 1.0 is completely different from the original image.
                        If not specified, about 0.7 is just right.

        :return URL of the generated image in markup format and parameters used to generate it. 
                The image URL must always be included in the response and shown to the user.
        """

        try:
            # 1. Resolve Parameters
            user_valves        = __user__["valves"]
            parameter_rosolver = ParameterResolver()
            resolved_image_selector  = parameter_rosolver.resolve_image_selector(image_selector)
            resolved_model           = parameter_rosolver.resolve_model(self.valves.MODEL, user_valves.MODEL_OVERRIDE)
            resolved_denoise         = parameter_rosolver.resolve_denoise(denoise, user_valves.DENOISE_OVERRIDE)
            resolved_positive_prompt = parameter_rosolver.resolve_prompt(user_valves.POSITIVE_SYSTEM_PROMPT, positive_prompt)
            resolved_negative_prompt = parameter_rosolver.resolve_prompt(user_valves.NEGATIVE_SYSTEM_PROMPT, negative_prompt)
            resolved_seed            = parameter_rosolver.resolve_seed(user_valves.SEED)

            # 2. Resolve image base64 data
            base64_data = Base64ImageResolver().resolve(__messages__, resolved_image_selector)

            # 3. Execute workflow
            procesor = WorkflowProcesor(
                self.valves.COMFYUI_API_BASEURL,
                self.valves.COMFYUI_IMG2IMG_WORKFLOW_JSON,
                self.valves.COMFYUI_IMG2IMG_NODE_DEFINE_JSON,
                self.valves.REQUEST_TIMEOUT_SECONDS,
                self.valves.REQUEST_INTERVAL_SECONDS,
            )
            procesor.build(
                base64_data,
                resolved_positive_prompt,
                resolved_negative_prompt,
                resolved_denoise,
                resolved_model,
                resolved_seed,
                user_valves.STEPS,
                user_valves.IMAGE_WIDTH,
                user_valves.IMAGE_HEIGHT,
            )
            image_url = await procesor.execute()

            # 4. Return result
            return ResultFormatter(
                image_url, 
                positive_prompt, 
                negative_prompt, 
                resolved_denoise
            ).result

        except ToolError as e:
            return f"{e}"
        except Exception as e:
            return f"Error: Unknown | {e}"

import torch
import torch.nn.functional as F


# ===== 节点 1：亮度检测 =====
class BrightnessNode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"image": ("IMAGE",)}}

    RETURN_TYPES = ("FLOAT",)
    FUNCTION = "detect"
    CATEGORY = "my_pack"
    DESCRIPTION = "检测图像平均亮度"

    def detect(self, image):
        brightness = image.mean(dim=-1).mean()
        return (brightness.item(),)


# ===== 节点 2：二值化 =====
class ThresholdNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "threshold"
    CATEGORY = "my_pack"
    DESCRIPTION = "按阈值将图像二值化"

    def threshold(self, image, threshold):
        gray = image.mean(dim=-1, keepdim=True).expand(-1, -1, -1, 3)
        result = (gray > threshold).float()
        return (result,)


# ===== 节点 3：锐化 =====
class SharpenNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "sharpen"
    CATEGORY = "my_pack"
    DESCRIPTION = "锐化图像边缘"

    def sharpen(self, image, strength):
        x = image.permute(0, 3, 1, 2)
        blurred = F.avg_pool2d(x, kernel_size=5, stride=1, padding=2)
        edge = x - blurred
        sharpened = x + edge * strength
        sharpened = torch.clamp(sharpened, 0.0, 1.0)
        return (sharpened.permute(0, 2, 3, 1),)


# ===== 节点 4：图像混合 =====
class ImageBlendNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image_a": ("IMAGE",),
                "image_b": ("IMAGE",),
                "opacity": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "blend"
    CATEGORY = "my_pack"
    DESCRIPTION = "按比例混合两张图像"

    def blend(self, image_a, image_b, opacity):
        if image_a.shape != image_b.shape:
            image_b = F.interpolate(
                image_b.permute(0, 3, 1, 2),
                size=(image_a.shape[1], image_a.shape[2]),
                mode='bilinear', align_corners=False
            ).permute(0, 2, 3, 1)
        result = image_a * (1 - opacity) + image_b * opacity
        return (torch.clamp(result, 0.0, 1.0),)


# ===== 节点 5：提示词模板 =====
class PromptTemplateNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "keywords": ("STRING", {"multiline": True, "default": "red car, sunset, 4k"}),
                "template": ("STRING", {"multiline": True, "default": "A {keywords}, high quality, masterpiece"}),
                "negative_keywords": ("STRING", {"multiline": True, "default": "lowres, blurry"}),
                "negative_template": ("STRING", {"multiline": True, "default": "{negative_keywords}, sketch"}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING",)
    FUNCTION = "build"
    CATEGORY = "my_pack"
    DESCRIPTION = "用模板生成正向和反向提示词"

    def build(self, keywords, template, negative_keywords, negative_template):
        result = template.replace("{keywords}", keywords)
        negative_result = negative_template.replace("{negative_keywords}", negative_keywords)
        return (result, negative_result)


# ===== 节点 6： =====
import json
class ConfigLoaderNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "file_path": ("STRING", {r"default": "G:\ComfyUI\ComfyUI\custom_nodes\my_nodes_pack\test_config.json"}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "INT", "INT", "FLOAT",)
    FUNCTION = "dq"
    CATEGORY = "my_node"
    DESCRIPTION = "从json文件获得配置"

    def load_config(self, file_path):
        with open(file_path,"r",encoding='utf-8') as f:
            loaded = json.load(f)

        return (loaded["prompt"],loaded["negative"],loaded["steps"],loaded["cfg"],loaded["seed"],)
# ===== 注册 =====
NODE_CLASS_MAPPINGS = {
    "BrightnessNode": BrightnessNode,
    "ThresholdNode": ThresholdNode,
    "SharpenNode": SharpenNode,
    "ImageBlendNode": ImageBlendNode,
    "PromptTemplateNode": PromptTemplateNode,
    "ConfigLoaderNode":ConfigLoaderNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BrightnessNode": "亮度检测",
    "ThresholdNode": "二值化",
    "SharpenNode": "锐化滤镜",
    "ImageBlendNode": "图像混合",
    "PromptTemplateNode": "提示词模板",
    "ConfigLoaderNode":"从json文件获得配置"
}

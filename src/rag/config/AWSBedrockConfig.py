from dataclasses import dataclass

@dataclass
class AWSBedrockConfig:
    model_id: str
    model_temperature: float
    max_tokens: int
    model_aws_region: str
    model_runtime: str
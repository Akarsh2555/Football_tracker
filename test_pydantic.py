from pydantic.v1 import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    test_var: 'Optional[int]' = None

print(Settings())

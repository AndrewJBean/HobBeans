from typing import Any, Type

try:
  from typing import Self
except ImportError:
  from typing_extensions import Self
import zipfile
import os

import pydantic


class BaseConfig(pydantic.BaseModel):
  """
  Base config class.
  """

  model_config = pydantic.ConfigDict(extra="forbid")

  def json_dumps(self: Self, *, exclude_none=True, indent=2, **kwargs) -> str:
    return self.model_dump_json(exclude_none=exclude_none, indent=indent, **kwargs)

  def json_dump_zip(self: Self, path: str, **kwargs) -> None:
    with zipfile.ZipFile(path, "w") as zf:
      zf.writestr("config.json", self.json_dumps(**kwargs))

  def json_dump(self: Self, path: str, **kwargs) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if path.endswith(".zip"):
      self.json_dump_zip(path, **kwargs)
    else:
      with open(path, "w") as f:
        f.write(self.json_dumps(**kwargs))

  @classmethod
  def json_loads(cls: Type[pydantic.BaseModel], s: str, **kwargs) -> Self:
    return cls.model_validate_json(s, strict=True, **kwargs)

  @classmethod
  def json_load_zip(cls, path: str, **kwargs) -> Self:
    with zipfile.ZipFile(path, "r") as zf:
      files_in_zip = zf.namelist()
      json_files = [f for f in files_in_zip if f.endswith(".json")]
      if len(json_files) != 1:
        raise ValueError(f"Expected exactly 1 json file in zip. Got: {json_files}")
      with zf.open(json_files[0]) as f:
        return cls.json_loads(f.read().decode(), **kwargs)

  @classmethod
  def json_load(cls, path: str, **kwargs) -> Self:
    if path.endswith(".zip"):
      return cls.json_load_zip(path, **kwargs)
    else:
      with open(path, "r") as f:
        return cls.json_loads(f.read(), **kwargs)


class UnionLikeConfig(BaseConfig):
  """
  Config for a union-like config, where exactly one
  field must be specified.
  """

  model_config = {"exclude_none": True}

  @pydantic.model_validator(mode="after")
  def _validate_fields(self: Self) -> Self:
    filled_fields = [f for f in self.model_fields.keys() if getattr(self, f) is not None]
    if len(filled_fields) != 1:
      raise ValueError(f"Exactly one field must be specified. Got: {filled_fields}")
    return self

  def which(self: Self) -> str:
    for f in self.model_fields.keys():
      if getattr(self, f) is not None:
        return f
    raise ValueError("No field is filled.")

  def unwrap(self: Self) -> Any:
    return getattr(self, self.which())

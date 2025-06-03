from typing import Dict, Union, List
from collections import defaultdict
from torch import nn
import torch
from torch.nn import Conv2d, Linear, AdaptiveAvgPool2d

from pointcept.utils.registry import Registry


MODELHOOKS = Registry("modelhook")


def build_model_hook(cfg):
    """Build modelhooks."""
    return MODELHOOKS.build(cfg)


@MODELHOOKS.register_module("ModelHook")
class BaseModelHook:
    class _DummyLogger:
        def info(self, msg, *args, **kwargs):
            pass

        def debug(self, msg, *args, **kwargs):
            pass

        def warning(self, msg, *args, **kwargs):
            pass

        def error(self, msg, *args, **kwargs):
            pass

        def exception(self, msg, *args, **kwargs):
            pass

    class _PrintLogger:
        def info(self, msg, *args, **kwargs):
            print("[INFO]", msg)

        def debug(self, msg, *args, **kwargs):
            print("[DEBUG]", msg)

        def warning(self, msg, *args, **kwargs):
            print("[WARNING]", msg)

        def error(self, msg, *args, **kwargs):
            print("[ERROR]", msg)

        def exception(self, msg, *args, **kwargs):
            print("[EXCEPTION]", msg)

    def __init__(
        self,
        hook_config: Dict[str, Union[List[str], str]],
        clone_tensor: bool = True,
        exclude_clone: Dict[str, List[str]] = None,
        logger=None,
    ):
        """
        Args:
            hook_config (dict): e.g., {"layer1": ["forward_input", "backward_outputGrad"]}
            clone_tensor (bool): Whether to clone the hooked outputs.
            exclude_clone (dict): Optional dict to exclude specific hooks from cloning.
        """
        self.clone_tensor = clone_tensor
        self.exclude_clone = exclude_clone or {}
        self.logger = logger or self._PrintLogger()
        self.model = None
        self.model_name = "Model"
        self.output = defaultdict(dict)
        self.hooks = defaultdict(dict)
        self.output_config = defaultdict(lambda: defaultdict(dict))
        self._hooked_modules = set()
        self._hooks_registered = False
        self.action_to_hook = {
            "inputGrad": "backward",
            "outputGrad": "backward",
            "input": "forward",
            "output": "forward",
        }
        for module_name, actions in hook_config.items():
            if isinstance(actions, str):
                actions = [actions]
            for action_str in actions:
                hook, action = action_str.split("_")
                assert hook == self.action_to_hook.get(
                    action
                ), f"Invalid hook-action combo: {action_str}"
                self.hooks[module_name][action_str] = None
                self.output[module_name][action_str] = None
                clone_flag = clone_tensor and not (
                    module_name in self.exclude_clone
                    and action_str in self.exclude_clone.get(module_name, [])
                )
                self.output_config[module_name][action_str]["clone"] = clone_flag

    def register_hooks(self, model):
        if not self._hooks_registered:
            self.model = getattr(model, "module", model)
            registered_module_names = set()
            for name, module in model.named_modules():
                if name in self.hooks and id(module) not in self._hooked_modules:
                    for action_str in self.hooks[name]:
                        hook, action = action_str.split("_")
                        hook_fn = self._make_closure(name, hook, action)
                        self.hooks[name][action_str] = getattr(
                            module, self._get_hook_register_fn(hook)
                        )(hook_fn)
                    self._hooked_modules.add(id(module))
                    registered_module_names.add(name)
            self.model_name = model.__class__.__name__
            self.logger.info(f"Hooks registered for {self.model_name}")
            self._hooks_registered = True
            # report unregistered modules
            configured_modules = set(self.hooks.keys())
            not_registered_modules = configured_modules - registered_module_names
            if not_registered_modules:
                self.logger.warning(
                    f"Hooks not registered for modules: {', '.join(not_registered_modules)}"
                )
        else:
            self.logger.warning(f"Hooks already registered for {self.model_name}")

    def remove_hooks(self):
        for module_name, hook_dict in self.hooks.items():
            for action_str, hook in hook_dict.items():
                if hook is not None:
                    hook.remove()
        self.logger.info(f"Hooks removed for {self.model_name}")

    def _make_closure(self, module_name, hook, action):
        clone_tensor = self.output_config[module_name][f"{hook}_{action}"]["clone"]

        def hook_fn(module, input, output):
            data = input if action == "input" else output
            self.output[module_name][f"{hook}_{action}"] = self._recursive(
                data, clone_tensor
            )

        def grad_fn(module, grad_input, grad_output):
            data = grad_input if action == "inputGrad" else grad_output
            self.output[module_name][f"{hook}_{action}"] = self._recursive(
                data, clone_tensor
            )

        return hook_fn if hook == "forward" else grad_fn

    def _recursive(self, obj, clone_tensor):
        if isinstance(obj, tuple):
            recur_obj = tuple(self._recursive(o, clone_tensor) for o in obj)
            if len(recur_obj) == 1:
                return recur_obj[0]
            else:
                return recur_obj
        elif isinstance(obj, torch.Tensor):
            return obj.clone() if clone_tensor else obj
        return obj

    def _get_hook_register_fn(self, hook_type: str):
        if hook_type == "forward":
            return "register_forward_hook"
        elif hook_type == "backward":
            return "register_full_backward_hook"
        else:
            raise ValueError("Invalid hook type")

    def __getitem__(self, key):
        return self.output[key]

    def __str__(self):
        indent = " " * 2
        s = f"Hooks of {self.model_name}: \n"
        for module_name, hooks in self.output.items():
            s += f"{indent}{module_name}:\n"
            for action, result in hooks.items():
                if isinstance(result, torch.Tensor):
                    s += f"{indent*2}{action}: {tuple(result.shape)}\n"
                elif isinstance(result, tuple):
                    for idx, r in enumerate(result):
                        shape = tuple(r.shape) if isinstance(r, torch.Tensor) else None
                        s += f"{indent*2}{action}_{idx}: {shape}\n"
                else:
                    s += f"{indent*2}{action}: {type(result)}\n"
        return s

    def __enter__(self):
        if self.model is None:
            raise RuntimeError("Model not set. Call `set_model(model)` first.")
        self.register_hooks(self.model)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.remove_hooks()
        if exc_type is KeyboardInterrupt and self.logger:
            self.logger.error("Program interrupted by keyboard input.")
            return True

    def set_logger(self, logger):
        self.logger = logger or self._DummyLogger()

    def set_model(self, model):
        self.model = getattr(model, "module", model)


# test Model
class TestModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv2d(in_channels=3, out_channels=32, kernel_size=3)
        self.layer1 = Linear(in_features=32, out_features=64)
        self.avgpool = AdaptiveAvgPool2d(1)

    def forward(self, x):
        conv1_out = self.conv1(x)
        avgpool_out = self.avgpool(conv1_out)
        flat = torch.flatten(avgpool_out, 1)
        layer1_out = self.layer1(flat)
        return {
            "conv1_out": conv1_out,
            "avgpool_out": avgpool_out,
            "flat": flat,
            "layer1_out": layer1_out,
        }


if __name__ == "__main__":
    print("start test")

    model = TestModel()
    print(model)

    model_hook = BaseModelHook({"layer1": "forward_output"})
    model_hook.register_hooks(model)

    model_hook_multi = BaseModelHook(
        {"conv1": ["forward_output"], "layer1": ["forward_input"]}
    )
    model_hook_multi.register_hooks(model)

    model_hook_multiAction = BaseModelHook(
        {
            "conv1": [
                "forward_input",
                "forward_output",
                "backward_inputGrad",
                "backward_outputGrad",
            ],
            "layer1": [
                "forward_input",
                "forward_output",
                "backward_inputGrad",
                "backward_outputGrad",
            ],
        }
    )
    model_hook_multiAction.register_hooks(model)

    model_hook_multiAction_with_config = BaseModelHook(
        {
            "conv1": [
                "forward_input",
                "forward_output",
                "backward_inputGrad",
                "backward_outputGrad",
            ],
            "layer1": [
                "forward_input",
                "forward_output",
                "backward_inputGrad",
                "backward_outputGrad",
            ],
        },
        exclude_clone={
            "conv1": ["forward_input", "forward_output"],
            "layer1": ["forward_input"],
        },
    )
    model_hook_multiAction_with_config.register_hooks(model)

    x = torch.randn([32, 3, 224, 224])
    model_out = model(x)
    loss = model_out["layer1_out"].sum()
    loss.backward()

    assert torch.equal(model_hook["layer1"]["forward_output"], model_out["layer1_out"])
    print(f"capture intermediate layer:\n{model_hook}")

    assert torch.equal(
        model_hook_multi["conv1"]["forward_output"], model_out["conv1_out"]
    )
    assert torch.equal(model_hook_multi["layer1"]["forward_input"], model_out["flat"])
    print(f"capture multi intermediate layer:\n{model_hook_multi}")

    print(
        f"capture multi intermediate layer with multi actions:\n{model_hook_multiAction}"
    )

    model_out["flat"][0] = -100000
    assert not torch.equal(
        model_hook_multi["layer1"]["forward_input"], model_out["flat"]
    )
    assert torch.equal(
        model_hook_multiAction_with_config["layer1"]["forward_input"],
        model_out["flat"],
    )

    model_hook.remove_hooks()
    model_hook_multi.remove_hooks()
    model_hook_multiAction.remove_hooks()
    model_hook_multiAction_with_config.remove_hooks()

    print("end of test, all passed")

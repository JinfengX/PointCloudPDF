from typing import Union, List
from collections import defaultdict
from torch import nn
import torch
from torch.nn import Conv2d, Linear, AdaptiveAvgPool2d

from .default import HookBase
from .builder import HOOKS


@HOOKS.register_module()
class ModelHook(HookBase):
    def __init__(
        self,
        register_module_name: Union[List[str], str],
        hook_and_action: Union[List[List[str]], List[str], str],
    ):
        """Initialize the ModelHook class.

        Args:
            register_module_name (Union[List[str], str]): The name of the module(s) to be registered. Can be a single module name or a list of module names.
            hook_and_action (Union[List[List[str]], List[str], str]): The type of hook to be registered and the corresponding action(s).
                If a single module is registered, hook_and_action should be a string indicating the type of hook and action to be registered.
                If multiple modules are being registered, hook_and_action should be a list of strings or sublists, where each sublist contains the type of hook and action(s) for each module.
                The hook types include 'forward' and 'backward', and the actions include 'getInput', 'getOutput', 'getInputGrad' and 'getOutputGrad'.
                Use '_' to connect the hook types and actions, e.g. 'forward_getInput', 'backward_getOutput'.
        """
        if isinstance(register_module_name, list) and isinstance(hook_and_action, list):
            assert len(register_module_name) == len(hook_and_action)
        else:
            assert isinstance(register_module_name, str)
            assert isinstance(hook_and_action, str)
            register_module_name = [register_module_name]
            hook_and_action = [hook_and_action]
        self.model_name = "Model"
        self.action_to_hook = {
            "getInputGrad": "backward",
            "getOutputGrad": "backward",
            "getInput": "forward",
            "getOutput": "forward",
        }
        self.output = defaultdict(lambda: defaultdict(lambda: None))
        self.hooks = defaultdict(lambda: defaultdict(lambda: None))
        for name, h_a in zip(register_module_name, hook_and_action):
            for h_a_item in self._parse_hook_and_action(h_a):
                hook, action = h_a_item.split("_")
                assert hook == self.action_to_hook.get(action, None)
                self.hooks[name][h_a_item] = None
                self.output[name][h_a_item] = None

    def register_hooks(self, model):
        for name, module in model.named_modules():
            if name in self.hooks.keys():
                for hook_action in self.hooks[name].keys():
                    hook, action = hook_action.split("_")
                    self.hooks[name][hook_action] = getattr(
                        module, self.hook_types(hook)
                    )(self.hook_closure(name, hook, action))
        self.model_name = model.__class__.__name__

    def remove_hooks(self):
        for module_name, hooks in self.hooks.items():
            for hook_action, hook_func in hooks.items():
                assert (
                    hook_func is not None
                ), f'Hook "{module_name}:{hook_action}" is not registered'
                hook_func.remove()

    def hook_closure(self, module_name, hook, action):
        if action in ["getInput", "getOutput"]:
            return self.get_feature_closure(module_name, hook, action)
        elif action in ["getInputGrad", "getOutputGrad"]:
            return self.get_grad_closure(module_name, hook, action)
        else:
            raise ValueError("Invalid hook action")

    def get_feature_closure(self, module_name, hook, action):
        def _get_feature(module, input, output):
            if action == "getInput":
                feature = input
            elif action == "getOutput":
                feature = output
            else:
                raise ValueError("Invalid action name")
            if isinstance(feature, tuple) and len(feature) == 1:
                self.output[module_name][f"{hook}_{action}"] = feature[0]
            else:
                self.output[module_name][f"{hook}_{action}"] = feature

        return _get_feature

    def get_grad_closure(self, module_name, hook, action):
        def _get_grad(module, grad_input, grad_output):
            if action == "getInputGrad":
                grad = grad_input
            elif action == "getOutputGrad":
                grad = grad_output
            else:
                raise ValueError("Invalid action name")
            if isinstance(grad, tuple) and len(grad) == 1:
                self.output[module_name][f"{hook}_{action}"] = grad[0]
            else:
                self.output[module_name][f"{hook}_{action}"] = grad

        return _get_grad

    @staticmethod
    def hook_types(name):
        if name == "forward":
            return "register_forward_hook"
        elif name == "backward":
            return "register_full_backward_hook"
        else:
            raise ValueError("Invalid hook type")

    def __getitem__(self, key):
        return self.output[key]

    def __str__(self):
        indent = " " * 2
        str_ = f"Hooks of {self.model_name}: \n"
        for module, hook_actions in self.output.items():
            str_ += f"{indent}{module}:\n"
            for h_a, output in hook_actions.items():
                if not isinstance(output, tuple):
                    show_output = None if output is None else output.shape
                    str_ += f"{indent*2}{h_a}: {show_output}\n"
                elif len(output) > 1:
                    for i, o in enumerate(output):
                        str_ += f"{indent*2}{h_a}_{i}: {o.shape if o is not None else None}\n"
        return str_

    def _parse_hook_and_action(self, h_a):
        if isinstance(h_a, list):
            return h_a
        elif isinstance(h_a, str):
            return [h_a]
        else:
            raise ValueError("Invalid hook action")


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

    model_hook = ModelHook("layer1", "forward_getOutput")
    model_hook.register_hooks(model)
    model_hook_multi = ModelHook(
        ["conv1", "layer1"], ["forward_getOutput", "forward_getInput"]
    )
    model_hook_multi.register_hooks(model)
    model_hook_multiAction = ModelHook(
        ["conv1", "layer1"],
        [
            [
                "forward_getInput",
                "forward_getOutput",
                "backward_getInputGrad",
                "backward_getOutputGrad",
            ],
            [
                "forward_getInput",
                "forward_getOutput",
                "backward_getInputGrad",
                "backward_getOutputGrad",
            ],
        ],
    )
    model_hook_multiAction.register_hooks(model)

    x = torch.randn([32, 3, 224, 224])
    model_out = model(x)
    loss = model_out["layer1_out"].sum()
    loss.backward()

    assert torch.equal(
        model_hook["layer1"]["forward_getOutput"], model_out["layer1_out"]
    )
    print(f"capture intermediate layer:\n{model_hook}")

    assert torch.equal(
        model_hook_multi["conv1"]["forward_getOutput"], model_out["conv1_out"]
    )
    assert torch.equal(
        model_hook_multi["layer1"]["forward_getInput"], model_out["flat"]
    )
    print(f"capture multi intermediate layer:\n{model_hook_multi}")

    print(
        f"capture multi intermediate layer with multi actions:\n{model_hook_multiAction}"
    )

    model_hook.remove_hooks()
    model_hook_multi.remove_hooks()
    model_hook_multiAction.remove_hooks()

    print("end of test")

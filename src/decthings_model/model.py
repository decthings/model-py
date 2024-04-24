import typing
import inspect
from decthings_api import DecthingsTensor

class DataLoaderBinary:
    def __init__(self, inner) -> None:
        pass

    def total_byte_size(self) -> int: # type: ignore
        pass

    def size(self) -> int: # type: ignore
        pass

    def shuffle(self) -> None:
        pass

    def shuffle_in_group(self, others: "list[DataLoaderBinary]") -> None:
        pass

    def position(self) -> int: # type: ignore
        pass

    def set_position(self, position: int) -> None: # noqa
        pass

    def remaining(self) -> int: # type: ignore
        pass

    def has_next(self, amount: int = 1) -> bool: # type: ignore
        pass

    async def next(self, amount: int = 1) -> "list[bytes]": # type: ignore
        pass

class DataLoader:
    def __init__(self, inner: DataLoaderBinary) -> None:
        self._inner = inner

    def total_byte_size(self) -> int:
        return self._inner.total_byte_size()

    def size(self) -> int:
        return self._inner.size()

    def shuffle(self) -> None:
        return self._inner.shuffle()

    def shuffle_in_group(self, others: "list[DataLoader]") -> None:
        if not isinstance(others, list) or any([not isinstance(x, DataLoader) for x in others]):
            raise TypeError(
                'DataLoader shuffle_in_group: Expected "others" to be a list of DataLoaders.'
            )
        return self._inner.shuffle_in_group([x._inner for x in others])

    def position(self) -> int:
        return self._inner.position()

    def set_position(self, position: int) -> None:
        self._inner.set_position(position)

    def remaining(self) -> int:
        return self._inner.remaining()

    def has_next(self, amount: int = 1) -> bool:
        return self._inner.has_next(amount)

    async def next(self, amount: int = 1) -> list[DecthingsTensor]:
        res = await self._inner.next(amount)
        return list(map(lambda x: DecthingsTensor.deserialize(x)[0], res))

class StateLoader:
    def __init__(self, inner) -> None:
        self._inner = inner

    def byte_size(self) -> int: # type: ignore
        pass

    async def read(self) -> bytes: # type: ignore
        pass

class TrainTracker:
    def __init__(self, inner) -> None:
        self._inner = inner

    def on_cancel(self, cb: typing.Callable):
        self._inner.on_cancel(cb)

    def failed(self, reason: str):
        self._inner.failed(reason)

    def metrics(self, metrics: "list[tuple[str, DecthingsTensor]]"):
        self._inner.metrics(list(map(lambda x: {"name": x[0], "value": x[1].serialize()}, metrics)))

    def progress(self, progress: "int | float"):
        self._inner.progress(progress)

DataLoaderMap = typing.Dict[str, DataLoader]
StateLoaderMap = typing.Dict[str, StateLoader]

class StateProvider:
    def provide(self, key: str, data: bytes) -> None:
        pass

    def provide_all(self, data: "list[dict]") -> None:
        pass

class _Model:
    @staticmethod
    def _create_data_loader_map(params: dict[str, DataLoaderBinary]) -> DataLoaderMap:
        new_params = {}
        for k in params.keys():
            new_params[k] = DataLoader(params[k])
        return new_params

    @staticmethod
    def createModelState(executor, params, provider):
        dataloader = _Model._create_data_loader_map(params)
        if isinstance(executor, dict):
            if "createModelState" not in executor:
                raise ValueError('The function "createModelState" was missing from the executor.')
            if not callable(executor["createModelState"]):
                raise ValueError(f'The property "createModelState" on the executor was not a function - got {str(type(executor["createModelState"]))}.')
            return executor["createModelState"](dataloader, provider)
        else:
            fn = getattr(executor, "createModelState", None)
            if fn is None:
                raise ValueError('The function "createModelState" was missing from the executor.')
            if not callable(fn):
                raise ValueError(f'The property "createModelState" on the executor was not a function - got {str(type(fn))}.')
            return executor.createModelState(dataloader, provider)

    @staticmethod
    async def instantiateModel(executor, model_state: bytes):
        if isinstance(executor, dict):
            if "instantiateModel" not in executor:
                raise ValueError('The function "instantiateModel" was missing from the executor.')
            if not callable(executor["instantiateModel"]):
                raise ValueError(f'The property "instantiateModel" on the executor was not a function - got {str(type(executor["instantiateModel"]))}.')
            instantiated = executor["instantiateModel"](model_state)
        else:
            fn = getattr(executor, "instantiateModel", None)
            if fn is None:
                raise ValueError('The function "instantiateModel" was missing from the executor.')
            if not callable(fn):
                raise ValueError(f'The property "instantiateModel" on the executor was not a function - got {str(type(fn))}.')
            instantiated = executor.instantiateModel(model_state)

        if inspect.isawaitable(instantiated):
            awaitedinstantiated = await instantiated
        else:
            awaitedinstantiated = instantiated

        return {
            "evaluate": lambda params: _Model.evaluate(awaitedinstantiated, params),
            "dispose": lambda: _Model.dispose(awaitedinstantiated),
            "getModelState": lambda provider: _Model.getModelState(awaitedinstantiated, provider),
            "train": lambda tracker, params: _Model.train(awaitedinstantiated, tracker, params)
        }

    @staticmethod
    async def evaluate(awaitedinstantiated, params):
        dataloader = _Model._create_data_loader_map(params)
        if isinstance(awaitedinstantiated, dict):
            if "evaluate" not in awaitedinstantiated:
                raise ValueError('The function "evaluate" was missing from the instantiated model.')
            if not callable(awaitedinstantiated["evaluate"]):
                raise ValueError(f'The property "evaluate" on the instantiated model was not a function - got {str(type(awaitedinstantiated["evaluate"]))}.')
            res = awaitedinstantiated['evaluate'](dataloader)
        else:
            fn = getattr(awaitedinstantiated, "evaluate", None)
            if fn is None:
                raise ValueError('The function "evaluate" was missing from the model.')
            if not callable(fn):
                raise ValueError(f'The property "evaluate" on the model was not a function - got {str(type(fn))}.')
            res = awaitedinstantiated.evaluate(dataloader)

        if inspect.isawaitable(res):
            awaitedres = await res
        else:
            awaitedres = res

        if not isinstance(awaitedres, list):
            raise ValueError(f'Evaluate: Expected return value of "evaluate" to be a list, not {str(type(awaitedres))}.')

        def map_fn(output_param):
            if not isinstance(output_param, dict):
                raise Exception('Evaluate: Expected each element in the return list of "evaluate" to be a dict.')
            if "data" not in output_param:
                raise ValueError('Evaluate: Expected each element in the return list of "evaluate" to contain a field "data".')
            if not isinstance(output_param["data"], list):
                raise ValueError(f'Evaluate: Expected the field "data" in each element of return list of "evaluate" to be a list, not {str(type(output_param["data"]))}.')
            def map_fn2(value):
                if not isinstance(value, DecthingsTensor):
                    raise ValueError(f'Evalutate: Expected each element in the list "data" in each element of return list of "evaluate" to be a DecthingsTensor, not {str(type(value))}.')
                return value.serialize()
            return {
                "name": output_param["name"],
                "data": list(map(map_fn2, output_param["data"]))
            }

        return list(map(map_fn, awaitedres))

    @staticmethod
    def dispose(awaitedinstantiated):
        if isinstance(awaitedinstantiated, dict):
            if "dispose" not in awaitedinstantiated:
                return
            if not callable(awaitedinstantiated["dispose"]):
                raise ValueError(f'The property "dispose" on the instantiated model was not a function - got {str(type(awaitedinstantiated["dispose"]))}.')
            return awaitedinstantiated["dispose"]()
        else:
            fn = getattr(awaitedinstantiated, "dispose", None)
            if fn is None:
                return
            if not callable(fn):
                raise ValueError(f'The property "dispose" on the instantiated model was not a function - got {str(type(fn))}.')
            return awaitedinstantiated.dispose()

    @staticmethod
    def getModelState(awaitedinstantiated, provider):
        if isinstance(awaitedinstantiated, dict):
            if "getModelState" not in awaitedinstantiated:
                raise ValueError('The function "getModelState" was missing from the instantiated model.')
            if not callable(awaitedinstantiated["getModelState"]):
                raise ValueError(f'The property "getModelState" on the instantiated model was not a function - got {str(type(awaitedinstantiated["getModelState"]))}.')
            return awaitedinstantiated["getModelState"](provider)
        else:
            fn = getattr(awaitedinstantiated, "getModelState", None)
            if fn is None:
                raise ValueError('The function "getModelState" was missing from the instantiated model.')
            if not callable(fn):
                raise ValueError(f'The property "getModelState" on the model was not a function - got {str(type(fn))}.')
            return awaitedinstantiated.getModelState(provider)

    @staticmethod
    def train(awaitedinstantiated, params, tracker):
        dataloader = _Model._create_data_loader_map(params)
        if isinstance(awaitedinstantiated, dict):
            if "train" not in awaitedinstantiated:
                raise ValueError('The function "train" was missing from the instantiated model.')
            if not callable(awaitedinstantiated["train"]):
                raise ValueError(f'The property "train" on the instantiated model was not a function - got {str(type(awaitedinstantiated["train"]))}.')
            return awaitedinstantiated["train"](dataloader, TrainTracker(tracker))
        else:
            fn = getattr(awaitedinstantiated, "train", None)
            if fn is None:
                raise ValueError('The function "train" was missing from the instantiated model.')
            if not callable(fn):
                raise ValueError(f'The property "train" on the instantiated model was not a function - got {str(type(fn))}.')
            return awaitedinstantiated.train(dataloader, TrainTracker(tracker))


def make_model(executor) -> dict:
    return {
        "createModelState": lambda params, provider: _Model.createModelState(executor, params, provider),
        "instantiateModel": lambda model_state: _Model.instantiateModel(executor, model_state)
    }

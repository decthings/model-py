import json
import sys
import os
import importlib.util
import asyncio
import traceback
import inspect
import typing

from .dataloader import createDataLoaderMap, createStateLoaderMap, createStateProvider, onDataProvided

runningProgram: typing.Any = None
instantiatedModels = {}
trainingSessions = {}

class TrainTracker:
    def __init__(self, id):
        self.id = id
        self._on_cancel = []
        self._complete = False

    def on_cancel(self, cb):
        self._on_cancel.append(cb)

    def progress(self, progress):
        if self._complete:
            raise Exception('TrainTracker progress: Cannot report progress after the training session was completed.')
        if not (type(progress) == int or type(progress) == float):
            raise Exception(f'TrainTracker progress: Invalid argument passed to progress(). Parameter "progress" must be an int or float, not {str(type(progress))}.')
        sendEventToParent('trainingProgress', { "trainingSessionId": self.id, "progress": progress }, [])

    def metrics(self, metrics):
        if self._complete:
            raise Exception('TrainTracker metrics: Cannot report metrics after the training session was completed.')
        if not isinstance(metrics, list):
            raise Exception(f'TrainTracker metrics: Invalid argument passed to metrics(). Parameter "metrics" must be a list, not {str(type(metrics))}.')
        if len(metrics) == 0:
            return
        nameslist = []
        byteslist = []
        for metric in metrics:
            if not isinstance(metric, dict):
                raise Exception('TrainTracker metrics: Invalid argument passed to metrics(). Expected each element of list "metrics" to be a dict.')
            if not "name" in metric or not isinstance(metric["name"], str):
                raise Exception('TrainTracker metrics: Invalid argument passed to metrics(). Expected each element of list "metrics" to contain a field "name" that is a string.')
            if not "value" in metric or not isinstance(metric["value"], bytes):
                raise Exception('TrainTracker metrics: Invalid argument passed to metrics(). Expected each element of list "metrics" to contain a field "value" that is bytes.')
            nameslist.append(metric["name"])
            byteslist.append(metric["value"])
        sendEventToParent('trainingMetrics', { "trainingSessionId": self.id, "names": nameslist }, byteslist)

def getErrorFromException():
    err_str = traceback.format_exc()
    if len(err_str) > 10000:
        err_str = err_str[0:10000] + f" (exception shortened - actual exception contained {len(err_str) - 10000} more characters)"
    return { "code": "exception", "details": err_str }

def initialize(args):
    sys.path.append(os.path.dirname(args["path"]) + "/_modules")
    sys.path.append(".")
    try:
        os.chdir(os.path.dirname(args["path"]))
        spec = importlib.util.spec_from_file_location("module.name", args["path"])
        if spec is None or spec.loader is None:
            raise Exception("Failed to load python module.")
        _module = importlib.util.module_from_spec(spec)
        if _module is None:
            raise Exception("Failed to load python module.")
        spec.loader.exec_module(_module)
        global runningProgram
        runningProgram = _module.model
    except:
        sendEventToParent("modelSessionInitialized", { "error": getErrorFromException() }, [])
        return
    sendEventToParent("modelSessionInitialized", {}, [])

async def callCreateModelState(args):
    complete = { "complete": False }
    map = createDataLoaderMap(complete, args["params"], sendDataEventToParent)
    provider = createStateProvider(complete, args["id"], sendEventToParent)
    other_model_states = {}
    for other_model_state in args["otherModelStates"]:
        other_model_states[other_model_state["id"]] = createStateLoaderMap(complete, other_model_state["state"], sendDataEventToParent)
    try:
        if isinstance(runningProgram, dict):
            res = runningProgram["createModelState"](map, provider)
        else:
            res = runningProgram.createModelState(map, provider)

        if inspect.isawaitable(res):
            await res
        
        complete["complete"] = True
        return { "result": {} }
    except:
        complete["complete"] = True
        return { "result": { "error": getErrorFromException() } }

async def callInstantiateModel(args):
    disposed = False

    modelFuture = asyncio.get_event_loop().create_future()

    def dispose():
        nonlocal disposed
        disposed = True
        del instantiatedModels[args["instantiatedModelId"]]
        modelFuture.set_result(None)

    stored = {
        "model": modelFuture,
        "dispose": dispose
    }

    instantiatedModels[args["instantiatedModelId"]] = stored

    complete = { "complete": False }
    stateLoaders = createStateLoaderMap(complete, args["state"], sendDataEventToParent)

    try:
        if isinstance(runningProgram, dict):
            res = runningProgram["instantiateModel"](stateLoaders)
        else:
            res = runningProgram.instantiateModel(stateLoaders)

        if inspect.isawaitable(res):
            awaitedres = await res
        else:
            awaitedres = res
        complete["complete"] = True
    except:
        complete["complete"] = True
        if not disposed:
            del instantiatedModels[args["instantiatedModelId"]]
            modelFuture.set_result(None)
        return { "result": { "error": getErrorFromException() } }

    stored["model"] = awaitedres

    def dispose2():
        if isinstance(awaitedres, dict):
            awaitedres["dispose"]()
        else:
            awaitedres.dispose()
    if disposed:
        dispose2()
    else:
        stored["dispose"] = dispose2
        modelFuture.set_result(awaitedres)
    return { "result": {} }

def callDisposeInstantiatedModel(args):
    if args["instantiatedModelId"] in instantiatedModels:
        instantiatedModel = instantiatedModels[args["instantiatedModelId"]]
        del instantiatedModels[args["instantiatedModelId"]]
        instantiatedModel["dispose"]

async def callTrain(args):
    tracker = TrainTracker(args["trainingSessionId"])
    trainingSessions[args["trainingSessionId"]] = tracker

    if not args["instantiatedModelId"] in instantiatedModels:
        return { "result": { "error": "instantiated_model_not_found" } }

    instantiatedModel = instantiatedModels[args["instantiatedModelId"]]
    if inspect.isawaitable(instantiatedModel["model"]):
        model = await instantiatedModel["model"]
        if model == None:
            return { "result": { "error": "instantiated_model_not_found" } }
    else:
        model = instantiatedModel["model"]

    complete = { "complete": False }
    map = createDataLoaderMap(complete, args["params"], sendDataEventToParent)
    try:
        if isinstance(model, dict):
            res = model["train"](map, tracker)
        else:
            res = model.train(map, tracker)

        if inspect.isawaitable(res):
            await res
        
        tracker._complete = True
        complete["complete"] = True
    except:
        tracker._complete = True
        complete["complete"] = True
        del trainingSessions[args["trainingSessionId"]]
        return { "result": { "error": getErrorFromException() } }

    del trainingSessions[args["trainingSessionId"]]
    return { "result": {} }

def callCancelTrain(args):
    if args["trainingSessionId"] in trainingSessions:
        tracker = trainingSessions[args["trainingSessionId"]]
        tracker.cancelled = True
        for cancel in tracker._on_cancel:
            cancel()

async def callEvaluate(args):
    if not args["instantiatedModelId"] in instantiatedModels:
        return { "result": { "error": "instantiated_model_not_found" } }
    instantiatedModel = instantiatedModels[args["instantiatedModelId"]]
    if inspect.isawaitable(instantiatedModel["model"]):
        model = await instantiatedModel["model"]
        if model == None:
            return { "result": { "error": "instantiated_model_not_found" } }
    else:
        model = instantiatedModel["model"]
    complete = { "complete": False }
    map = createDataLoaderMap(complete, args["params"], sendDataEventToParent)
    try:
        if isinstance(model, dict):
            res = model["evaluate"](map)
        else:
            res = model.evaluate(map)

        if inspect.isawaitable(res):
            awaitedres = await res
        else:
            awaitedres = res
        
        complete["complete"] = True

        outputs = []
        alsoSend = []

        if not isinstance(awaitedres, list):
            raise Exception('Evaluate: Expected return value of "evaluate" to be a list.')
        for el in awaitedres:
            if not isinstance(el, dict):
                raise Exception('Evaluate: Expected each element in the return list of "evaluate" to be a dict.')
            if not "name" in el or not isinstance(el['name'], str):
                raise Exception('Evaluate: Expected each element in the return list of "evaluate" to contain a field "name" that is a string.')
            if not "data" in el or not isinstance(el['data'], list):
                raise Exception('Evaluate: Expected each element in the return list of "evaluate" to contain a field "data" that is a list.')
            for el2 in el['data']:
                if not isinstance(el2, bytes):
                    raise Exception('Evaluate: Expected each element in the return list of "evaluate" to contain a field "data" that is a list where each element is a bytes object. Got something other than bytes.')
            outputs.append({ "name": el['name'], "byteSizes": [len(x) for x in el['data']] })
            alsoSend.extend(el['data'])
    except:
        complete["complete"] = True
        return { "result": { "error": getErrorFromException() } }

    return { "result": { "outputs": outputs }, "alsoSend": [b''.join(alsoSend)] }

async def callGetModelState(args):
    if not args["instantiatedModelId"] in instantiatedModels:
        return { "result": { "error": "instantiated_model_not_found" } }
    instantiatedModel = instantiatedModels[args["instantiatedModelId"]]
    if inspect.isawaitable(instantiatedModel["model"]):
        model = await instantiatedModel["model"]
        if model == None:
            return { "result": { "error": "instantiated_model_not_found" } }
    else:
        model = instantiatedModel["model"]
    
    complete = { "complete": False }
    provider = createStateProvider(complete, args["id"], sendEventToParent)

    try:
        if isinstance(model, dict):
            res = model["getModelState"](provider)
        else:
            res = model.getModelState(provider)

        if inspect.isawaitable(res):
            await res
        
        complete["complete"] = True
    except:
        complete["complete"] = True
        return { "result": { "error": getErrorFromException() } }

    return { "result": {} }

rpc = {
    "initialize": initialize,
    "callCreateModelState": callCreateModelState,
    "callInstantiateModel": callInstantiateModel,
    "callDisposeInstantiatedModel": callDisposeInstantiatedModel,
    "callTrain": callTrain,
    "callEvaluate": callEvaluate,
    "callGetModelState": callGetModelState,
}

sock_lock = asyncio.Lock()
sock: typing.Any = None

async def _sendMessageToParent(message):
    global sock
    async with sock_lock:
        sock.write(message)
        await sock.drain()

def sendMessageToParent(message):
    loop = asyncio.get_event_loop()
    loop.create_task(_sendMessageToParent(message))

def sendEventToParent(event, params, additionalSegments):
    toSend = [json.dumps({ "event": event, "params": params }).encode(), *additionalSegments]

    argsTotalSize = 0
    for el in toSend:
        argsTotalSize += len(el)

    finalBuffer = bytearray(6 + len(toSend) * 8 + argsTotalSize)
    finalBuffer[0] = 0
    finalBuffer[1:5] = (len(toSend) - 1).to_bytes(4, 'big')

    position = 5
    for el in toSend:
        finalBuffer[position : position + 8] = len(el).to_bytes(8, 'big')
        finalBuffer[position + 8 : position + 8 + len(el)] = el
        position += 8 + len(el)

    finalBuffer[position] = 0

    sendMessageToParent(finalBuffer)

def sendDataEventToParent(event):
    encoded = json.dumps(event).encode()
    sendMessageToParent((1).to_bytes(1, 'big') + len(encoded).to_bytes(8, 'big') + encoded)

async def processMessage(buf):
    parsed = json.loads(buf.decode('utf-8'))
    
    response = rpc[parsed["method"]](parsed["params"])

    _response = None
    if asyncio.iscoroutine(response):
        _response = await response
    else:
        _response = response

    if not "id" in parsed["params"]:
        return

    toSend = [json.dumps({ "id": parsed["params"]["id"], "result": _response["result"] }).encode(), *(_response['alsoSend'] if 'alsoSend' in _response else [])]

    argsTotalSize = 0
    for el in toSend:
        argsTotalSize += len(el)

    finalBuffer = bytearray(6 + len(toSend) * 8 + argsTotalSize)
    finalBuffer[0] = 0
    finalBuffer[1:5] = (len(toSend) - 1).to_bytes(4, 'big')

    position = 5
    for el in toSend:
        finalBuffer[position : position + 8] = len(el).to_bytes(8, 'big')
        finalBuffer[position + 8 : position + 8 + len(el)] = el
        position += 8 + len(el)

    finalBuffer[position] = 0

    sendMessageToParent(finalBuffer)

async def inner_main():
    global sock
    global buffer

    reader, writer = await asyncio.open_unix_connection(path = os.environ.get('IPC_PATH'))
    sock = writer

    while True:
        first_byte = (await reader.readexactly(1))[0]
        if first_byte == 0:
            # RPC
            segment_len = int.from_bytes(await reader.readexactly(8), 'big')
            segment_data = await reader.readexactly(segment_len)
            loop = asyncio.get_event_loop()
            loop.create_task(processMessage(segment_data))
        else:
            # Provide data
            request_id = int.from_bytes(await reader.readexactly(4), 'big')
            num_segments = int.from_bytes(await reader.readexactly(4), 'big')
            data = []
            for i in range(num_segments):
                segment_len = int.from_bytes(await reader.readexactly(8), 'big')
                segment_data = await reader.readexactly(segment_len)
                data.append(segment_data)
            onDataProvided(request_id, data)

def main():
    loop = asyncio.new_event_loop()
    loop.run_until_complete(inner_main())

import asyncio
import math

class DataLoader:
    def __init__(self, inner):
        self._inner = inner
        self._position = 0

    def total_byte_size(self):
        return self._inner["totalByteSize"]

    def shuffle(self):
        self._inner["shuffle"]([self._inner["dataset"]])

    def shuffle_in_group(self, others: "list[DataLoader]"):
        if not isinstance(others, list) or any([not isinstance(x, DataLoader) for x in others]):
            raise TypeError(
                'DataLoader shuffle_in_group: Expected "others" to be a list of DataLoaders.'
            )
        self._inner["shuffle"]([self._inner["dataset"], *[x._inner["dataset"] for x in others]])

    def size(self) -> int:
        return self._inner["size"]

    def position(self) -> int:
        return self._position

    def set_position(self, position: "int | float"):
        if not isinstance(position, int) and not isinstance(position, float):
            raise TypeError(
                'DataLoader set_position: Expected "position" to be of type "int" or "float".'
            )
        position = max(math.floor(position), 0)
        if position >= self.size():
            raise ValueError(
                'DataLoader set_position: Cannot set a position which is greater than or equal to the data size. The data size was ' +
                str(self._inner["size"]) + ', and position ' +
                str(self._position) + ' was attempted to be set.'
            )
        self._position = position

    def remaining(self) -> int:
        return self._inner["size"] - self._position

    def has_next(self, amount = 1) -> bool:
        if not isinstance(amount, int) and not isinstance(amount, float):
            raise TypeError(
                'DataLoader has_next: Expected "amount" to be of type "int" or "float".'
            )
        return self.remaining() >= math.floor(amount)

    async def next(self, amount = 1):
        if not isinstance(amount, int) and not isinstance(amount, float):
            raise TypeError(
                'DataLoader next: Expected "amount" to be of type "int" or "float".'
            )
        numToRead = min(math.floor(amount), self.remaining())

        if numToRead <= 0:
            return []

        ret = self._inner["read"](self._position, numToRead)

        self._position += numToRead

        return await ret

class WeightsLoader:
    def __init__(self, inner: DataLoader):
        self._inner = inner

    def byte_size(self):
        return self._inner.total_byte_size()

    async def read(self):
        self._inner.set_position(0)
        return (await self._inner.next())[0]


waiting = {}
dataRequestIdCounter = 0

async def read(dataset, startIndex, amount, sendDataEventToParent):
    global dataRequestIdCounter
    global waiting

    fut = asyncio.get_event_loop().create_future()
    
    def resolve(val):
        fut.set_result(val)
    
    reqId = dataRequestIdCounter
    dataRequestIdCounter += 1
    waiting[reqId] = resolve
    
    sendDataEventToParent({ "event": "requestData", "requestId": reqId, "dataset": dataset, "startIndex": startIndex, "amount": amount })
    
    return await fut

def createDataLoader(complete, dataset, size, total_byte_size, sendDataEventToParent):
    async def inner_read(startIndex, amount):
        if complete["complete"]:
            raise Exception('DataLoader read: Cannot read data after the function was completed.')
        return await read(dataset, startIndex, amount, sendDataEventToParent)
    
    def shuffle(datasets):
        sendDataEventToParent({ "event": "shuffle", "datasets": datasets })

    return DataLoader({
        "dataset": dataset,
        "size": size,
        "totalByteSize": total_byte_size,
        "read": inner_read,
        "shuffle": shuffle,
    })

def createDataLoaderMap(complete, params, sendDataEventToParent):
    map = {}
    for param in params:
        map[param["name"]] = createDataLoader(complete, param['dataset'], param['amount'], param['totalByteSize'], sendDataEventToParent)
    return map

def createWeightsLoaderMap(complete, params, sendDataEventToParent):
    map = {}
    for param in params:
        map[param["name"]] = WeightsLoader(createDataLoader(complete, param['dataset'], param['amount'], param['totalByteSize'], sendDataEventToParent))
    return map

class WeightsProvider:
    def __init__(self, provide):
        self._provide = provide

    def provide_all(self, data):
        if not isinstance(data, list) or any([not isinstance(x, dict) or not 'key' in x or not 'data' in x or not isinstance(x['key'], str) or not isinstance(x['data'], bytes) for x in data]):
            raise TypeError('WeightsProvider provide: Expected "data" to be a list of dictionaries like { "key": str, "data": bytes }.')
        self._provide(data)

    def provide(self, key, data):
        if not isinstance(key, str):
            raise TypeError('WeightsProvider provide: Expected "key" to be a string.')
        if not isinstance(data, bytes):
            raise TypeError('WeightsProvider provide: Expected "data" to be bytes.')
        self._provide([{ "key": key, "data": data }])


def createWeightsProvider(complete, commandId, sendEventToParent):
    provided = set()
    def _provide(data):
        if complete["complete"]:
            raise Exception('WeightsProvider: Cannot provide data after the function was completed.')

        for x in data:
            if x['key'] in provided:
                raise Exception(f'WeightsProvider: Weight key "{x["key"]}" was provided multiple times.')
            provided.add(x['key'])

        if len(provided) + len(data) > 100:
            raise Exception('WeightsProvider: Cannot provide more than 100 keys.')
        for el in data:
            if len(el) > math.pow(1024, 3):
                raise TypeError('WeightsProvider: Cannot provide more than 1 gigabyte for a single key. Split it into multiple keys.')

        i = 0
        while i < len(data):
            to_send = [data[i]['data']]
            names = [data[i]['key']]
            i += 1
            total_length = len(to_send[0])
            while i < len(data) and len(data[i]['data']) + total_length < math.pow(1024, 3):
                total_length += len(data[i]['data'])
                to_send.append(data[i]['data'])
                names.append(data[i]['key'])
                i += 1
            sendEventToParent('provideWeightData', { "commandId": commandId, "names": names }, to_send)

    return WeightsProvider(_provide)


def onDataProvided(requestId, data):
    if not requestId in waiting:
        return

    _waiting = waiting[requestId]
    del waiting[requestId]

    _waiting(data)

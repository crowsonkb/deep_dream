from collections import namedtuple

import numpy as np

import deep_dream as dd

TileRequest = namedtuple('TileRequest', 'resp data layers kwargs')
TileResponse = namedtuple('TileResponse', 'resp grad')


class TileWorker:
    def __init__(self, req_q, resp_q, cnndata, gpu=None):
        self.req_q = req_q
        self.resp_q = resp_q
        self.proc = dd.CTX.Process(target=self._run, args=(cnndata, gpu), daemon=True)
        self.proc.start()

    def __del__(self):
        self.proc.terminate()

    # pylint: disable=attribute-defined-outside-init
    def _run(self, cnndata, gpu=None):
        import caffe

        self.net = caffe.Classifier(str(cnndata.deploy), str(cnndata.model),
                                    mean=np.float32(cnndata.mean), channel_swap=(2, 1, 0))
        self.data = dd._LayerIndexer(self.net, 'data')
        self.diff = dd._LayerIndexer(self.net, 'diff')

        if gpu is not None:
            caffe.set_device(gpu)
            caffe.set_mode_gpu()
        else:
            caffe.set_mode_cpu()

        while True:
            req = self.req_q.get()
            grad = self._grad_single_tile(req.data, req.layers, **req.kwargs)
            resp = TileResponse(req.resp, grad)
            self.resp_q.put(resp)
            self.req_q.task_done()

    def _grad_single_tile(self, data, layers, auto_weight=True):
        self.net.blobs['data'].reshape(1, 3, data.shape[1], data.shape[2])
        self.data['data'] = data

        for layer in layers.keys():
            self.diff[layer] = 0
        self.net.forward(end=next(iter(layers.keys())))
        layers_list = list(layers.keys())
        for i, layer in enumerate(layers_list):
            if auto_weight:
                self.diff[layer] += \
                    self.data[layer]*layers[layer]/np.abs(self.data[layer]).sum()
            else:
                self.diff[layer] += self.data[layer]*layers[layer]
            if i+1 == len(layers):
                self.net.backward(start=layer)
            else:
                self.net.backward(start=layer, end=layers_list[i+1])

        return self.diff['data']

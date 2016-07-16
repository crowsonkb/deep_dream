from collections import namedtuple
import os

import numpy as np

import deep_dream as dd

TileRequest = namedtuple('TileRequest', 'resp data layers')
TileResponse = namedtuple('TileResponse', 'resp grad obj denom')


class TileWorker:
    def __init__(self, req_q, resp_q, cnndata, gpu=None):
        self.req_q = req_q
        self.resp_q = resp_q
        if gpu is not None:
            name = 'GPU-%d' % gpu
        else:
            name = 'CPU'
        self.proc = dd.CTX.Process(target=self._run, args=(cnndata, gpu), name=name, daemon=True)
        self.proc.start()

    def __del__(self):
        self.proc.terminate()

    # pylint: disable=attribute-defined-outside-init
    def _run(self, cnndata, gpu=None):
        if gpu is not None:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
        import caffe

        self.net = caffe.Classifier(str(cnndata.deploy), str(cnndata.model),
                                    mean=np.float32(cnndata.mean), channel_swap=(2, 1, 0))
        self.data = dd._LayerIndexer(self.net, 'data')
        self.diff = dd._LayerIndexer(self.net, 'diff')

        if gpu is not None:
            caffe.set_mode_gpu()
        else:
            caffe.set_mode_cpu()

        while True:
            for blob in self.net.blobs:
                self.diff[blob] = 0
            req = self.req_q.get()
            grad, obj = self._grad_single_tile(req.data, req.layers)
            resp = TileResponse(req.resp, grad, obj[0], obj[1])
            self.resp_q.put(resp)
            self.req_q.task_done()

    def _grad_single_tile(self, data, layers):
        obj_num, obj_denom = 0, 0
        self.net.blobs['data'].reshape(1, 3, data.shape[1], data.shape[2])
        self.data['data'] = data

        layers_list = list(layers.keys())
        self.net.forward(end=layers_list[0])
        for i, layer in enumerate(layers_list):
            weighted = self.data[layer] * layers[layer]

            obj_num += dd.normf(weighted, 1)
            if not np.ndim(layers[layer]):
                obj_denom += self.data[layer].size * layers[layer]
            else:
                obj_denom += self.data[layer][0].size * dd.normf(layers[layer], 1)

            self.diff[layer] += weighted / (dd.normf(weighted)+dd.EPS)

            if i+1 == len(layers):
                self.net.backward(start=layer)
            else:
                self.net.backward(start=layer, end=layers_list[i+1])

        return self.diff['data'].copy(), (obj_num, obj_denom)

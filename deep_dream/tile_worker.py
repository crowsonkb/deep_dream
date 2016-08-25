from collections import namedtuple
import logging
import os

import deep_dream as dd

logger = logging.getLogger(__name__)

TileRequest = namedtuple('TileRequest', 'resp data layers cleanup guide_mode')
TileResponse = namedtuple('TileResponse', 'resp grad')


class TileWorker:
    def __init__(self, req_q, resp_q, cnndata, gpu=None):
        self.req_q = req_q
        self.resp_q = resp_q
        if gpu is not None:
            name = 'GPU%d' % gpu
        else:
            name = 'CPU'
        self.proc = dd.CTX.Process(target=self._run, args=(cnndata, gpu), name=name, daemon=True)
        self.proc.start()
        logger.debug('Started tile worker (%s, pid %d)', name, self.proc.pid)

    def __del__(self):
        if not self.proc.exitcode:
            self.proc.terminate()
            logger.debug('Shut down tile worker (%s, pid %d)', self.proc.name, self.proc.pid)

    # pylint: disable=attribute-defined-outside-init
    def _run(self, cnndata, gpu=None):
        if gpu is not None:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
        import caffe
        caffe.set_random_seed(0)

        self.net = caffe.Net(str(cnndata.deploy), 1, weights=str(cnndata.model))
        self.data = dd._LayerIndexer(self.net, 'data')
        self.diff = dd._LayerIndexer(self.net, 'diff')

        if gpu is not None:
            caffe.set_mode_gpu()
        else:
            caffe.set_mode_cpu()

        while True:
            req = self.req_q.get()
            if req.cleanup:
                for blob in self.net.blobs:
                    self.diff[blob] = 0
            grad = self._grad_single_tile(req.data, req.layers, guide_mode=req.guide_mode)
            resp = TileResponse(req.resp, grad)
            self.resp_q.put(resp)
            self.req_q.task_done()

    def _grad_single_tile(self, data, layers, guide_mode=1):
        self.net.blobs['data'].reshape(1, 3, data.shape[1], data.shape[2])
        self.data['data'] = data

        layers_list = list(layers.keys())
        for layer in layers_list:
            self.diff[layer] = 0
        self.net.forward(end=layers_list[0])
        for i, layer in enumerate(layers_list):
            if not guide_mode:
                weighted = self.data[layer] * layers[layer]
                self.diff[layer] += weighted / (dd.normf(weighted)+dd.EPS)
            else:
                assert layers[layer].ndim == 3
                cur_feat = self.data[layer].reshape(self.data[layer].shape[0], -1)
                guide_feat = layers[layer].reshape(layers[layer].shape[0], -1)
                a = cur_feat.T @ guide_feat
                self.diff[layer].reshape(cur_feat.shape[0], -1)[:] += guide_feat[:, a.argmax(1)]

            if i+1 == len(layers):
                self.net.backward(start=layer)
            else:
                self.net.backward(start=layer, end=layers_list[i+1])

        return self.diff['data'].copy()

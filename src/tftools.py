import models.cnn as cnn

base_cnn_build_fun = cnn.build_cnn_network
cnn_override = cnn.build_cnn_network
def _cnn_override(*args, **kwargs):
    global cnn_override
    return cnn_override(*args, **kwargs)

cnn.build_cnn_network = _cnn_override


def override_cnn(filters: list, kernels: list, strides: list):
    global cnn_override
    def build_cnn_network(*args, **kwargs):
        return base_cnn_build_fun(
            filters=filters,
            kernels=kernels,
            strides=strides
        )
    cnn_override = build_cnn_network

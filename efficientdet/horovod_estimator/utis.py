import socket

try:
    import horovod.tensorflow as hvd
except ImportError:
    hvd = None
import tensorflow as tf
from tensorflow.core.framework import summary_pb2
from tensorflow.core.framework.summary_pb2 import Summary
from tensorflow.core.framework.tensor_pb2 import TensorProto
from tensorflow.core.framework.tensor_shape_pb2 import TensorShapeProto

def is_rank0():
    if hvd is not None:
        return hvd.rank() == 0
    else:
        True


global IS_HVD_INIT
IS_HVD_INIT = False


def hvd_try_init():
    global IS_HVD_INIT
    if not IS_HVD_INIT and hvd is not None:
        hvd.init()
        IS_HVD_INIT = True
        if tf.__version__.startswith('1.13'):
            tf.get_logger().propagate = False
        else:
            from tensorflow.python.platform import tf_logging
            tf_logging._get_logger().propagate = False

        if hvd.rank() == 0:
            tf.logging.set_verbosity('INFO')
        else:
            tf.logging.set_verbosity('WARN')


def hvd_info(msg):
    hvd_try_init()
    if hvd is not None:
        head = 'hvd rank{}/{} in {}'.format(hvd.rank(), hvd.size(), socket.gethostname())
    else:
        head = '{}'.format(socket.gethostname())
    tf.logging.info('{}: {}'.format(head, msg))


def hvd_info_rank0(msg, with_head=True):
    hvd_try_init()
    if is_rank0():
        if with_head:
            if hvd is not None:
                head = 'hvd only rank{}/{} in {}'.format(hvd.rank(), hvd.size(), socket.gethostname())
            else:
                head = '{}'.format(socket.gethostname())
            tf.logging.info('{}: {}'.format(head, msg))
        else:
            tf.logging.info(msg)


# to-do
def writer_text_summary(tag, text, model_dir):
    summary_metadata = summary_pb2.SummaryMetadata(
        plugin_data=summary_pb2.SummaryMetadata.PluginData(plugin_name='text'))
    tensor = TensorProto(dtype='DT_STRING', string_val=[text.encode(encoding='utf_8')],
                         tensor_shape=TensorShapeProto(dim=[TensorShapeProto.Dim(size=1)]))
    summary = Summary(value=[Summary.Value(node_name=tag, metadata=summary_metadata, tensor=tensor)])
    writer = tf.summary.FileWriterCache.get(model_dir)
    writer.add_summary(summary)
    writer.flush()

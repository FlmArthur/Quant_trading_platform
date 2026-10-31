import os
import tempfile

_data=tempfile.TemporaryDirectory(prefix='quant-tests-')
os.environ['QUANT_DATA_DIR']=_data.name

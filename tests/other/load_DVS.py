
import os
from snntoolbox.io_utils.AedatTools import ImportAedat

path = '/home/rbodo/.snntoolbox/Datasets/roshambo/DVS_all/paper'
datafile = os.path.join(path, 'paper_enea_back.aedat')
args = {'filePathAndName': datafile, 'dataTypes': {'polarity'}}
output = ImportAedat.import_aedat(args)
timestamps_all = output['data']['polarity']['timeStamp']
xaddr_all = output['data']['polarity']['x']
yaddr_all = output['data']['polarity']['y']
pol_all = output['data']['polarity']['polarity']

print(pol_all)

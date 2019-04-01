data_path = '/home/tri/skripsi/dataset/'
filename = '000001-1'

df = open(data_path + 'camera-intrinsics/' + filename + '.txt')
data = []
for i in range(3):
    line = df.readline().split('\t')
    data.append([float(a) for a in line[:3]])
print(data)

df = open(data_path + 'camera-intrinsics/' + filename + '.txt')
d = [[float(a) for a in line.split('\t')[:3]] for line in df.readlines()[:3]]
print(d)
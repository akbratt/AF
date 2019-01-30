import torch.nn as nn
import torch.nn.functional as F
import torch

d = 0.0

class Concat(nn.Module):
    def __init__(self):
        super(Concat, self).__init__()

    def forward(self, x1, x2):
        cat = torch.cat([x1,x2], dim=1)
        return cat

class EncoderNorm_2d(nn.Module):
    def __init__(self, channels):
        super(EncoderNorm_2d, self).__init__()
        self.bn = nn.InstanceNorm2d(channels,track_running_stats=True)
        # self.bn = nn.BatchNorm2d(channels)

    def forward(self, x):
        return self.bn(x)

class Res_Down(nn.Module):
    def __init__(self, in_chan, out_chan):
        super(Res_Down, self).__init__()

        self.conv1 = nn.Conv2d(in_chan, out_chan, 4, stride = (2,2),\
                padding=(1,1))
        self.bn1 = EncoderNorm_2d(out_chan)
        self.drop = nn.Dropout2d(d)

        self.conv2 = nn.Conv2d(out_chan, out_chan, 3, padding=(1,1))
        self.bn2 = EncoderNorm_2d(out_chan)


    def forward(self, x):
        x = self.drop(F.relu(self.bn1(self.conv1(x))))
        out = self.bn2(self.conv2(x))

        return F.relu(x + out)

class Res_Up(nn.Module):
    def __init__(self, in_chan, out_chan):
        super(Res_Up, self).__init__()

        self.conv1 = nn.ConvTranspose2d(in_chan, out_chan, 4, stride = (2,2),\
                padding=(1,1))
        self.bn1 = EncoderNorm_2d(out_chan)
        self.drop = nn.Dropout2d(d)

        self.conv2 = nn.Conv2d(out_chan, out_chan, 3, padding=(1,1))
        self.bn2 = EncoderNorm_2d(out_chan)


    def forward(self, x):
        x = self.drop(F.relu(self.bn1(self.conv1(x))))
        out = self.bn2(self.conv2(x))

        return F.relu(x + out)

class Res_Final(nn.Module):
    def __init__(self, in_chan, out_chan):
        super(Res_Final, self).__init__()

        self.conv1 = nn.Conv2d(in_chan, in_chan, 3, padding=(1,1))
        self.bn1 = EncoderNorm_2d(in_chan)
        self.drop = nn.Dropout2d(d)

        self.conv2 = nn.Conv2d(in_chan, out_chan, 3, padding=(1,1))


    def forward(self, x):
        out = self.drop(F.relu(self.bn1(self.conv1(x))))
        out = self.conv2(x + out)

        return out


class Net(nn.Module):

    def __init__(self, outs):
        super(Net, self).__init__()

        self.cat = Concat()

        self.conv1 = nn.Conv2d(1, 64, 3, padding=(1,1))
        self.bn1 = EncoderNorm_2d(64)

        self.drop = nn.Dropout2d(d)

        self.conv2 = nn.Conv2d(64, 64, 3, padding=(1,1))
        self.bn2 = EncoderNorm_2d(64)

        self.Rd1 = Res_Down(64, 128)
        self.Rd2 = Res_Down(128, 256)
        self.Rd3 = Res_Down(256, 512)
        self.Rd4 = Res_Down(512, 512)

        self.fudge = nn.ConvTranspose2d(512, 256, 4, stride = (2,2),\
                padding = (1,1))

        self.Ru3 = Res_Up(512,512)
        self.Ru2 = Res_Up(512,256)
        self.Ru1 = Res_Up(256,128)
        self.Ru0 = Res_Up(128,64)

        self.Rf = Res_Final(64,outs)


    def forward(self, x):
        out = self.drop(F.relu(self.bn1(self.conv1(x))))
        e0 = F.relu(self.bn2(self.conv2(out)))

        e1 = self.Rd1(e0)
        e2 = self.Rd2(e1)
        e3 = self.Rd3(e2)
        e4 = self.Rd4(e3)

        d3 = self.Ru3(e4)
        d2 = self.Ru2(self.cat(d3[:,256:],e3[:,256:]))
        d1 = self.Ru1(self.cat(d2[:,128:],e2[:,128:]))
        d0 = self.Ru0(self.cat(d1[:,64:],e1[:,64:]))

        out = self.Rf(self.cat(e0[:,32:],d0[:,32:]))

        return out

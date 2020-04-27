import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 32, kernel_size = 5)
		self.pool = nn.MaxPool2d(2)
		self.drp1 = nn.Dropout(0.3)
		self.conv2 = nn.Conv2d(in_channels = 32, out_channels = 128, kernel_size = 3, padding = 2)
		self.drp2 = nn.Dropout(0.3)
		self.conv3 = nn.Conv2d(in_channels = 128, out_channels = 252, kernel_size = 3, padding = 2)
		self.drp2 = nn.Dropout(0.3)
		self.bn1 = nn.BatchNorm2d(32)
		self.bn2 = nn.BatchNorm2d(128)
		self.bn_2 = nn.BatchNorm2d(252)
		# self.fc1 = nn.Linear(252 * 5 * 5, 512)
		# self.bn3 = nn.BatchNorm1d(512)
		self.fc2 = nn.Linear(252 * 5 * 5, 252)
		self.bn4 = nn.BatchNorm1d(252)
		self.fc3 = nn.Linear(252, 10)

	def forward(self, x):
		x1 = self.conv1(x)
		x = F.relu(self.pool(self.bn1(self.conv1(x))))
		x = self.drp1(x)
		x = F.relu(self.pool(self.bn2(self.conv2(x))))
		x = self.drp2(x)
		x = F.relu(self.pool(self.bn_2(self.conv3(x))))
		x = self.drp2(x)
		x = x.view(-1, 252 * 5 * 5)
		# x = F.relu(self.bn3(self.fc1(x)))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		
		return x,x1 


# importing all the required libraries
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt

import argparse
import model_CNN2 as mdl
import model_CNN3 as mdl1

#Defining Train function for training the model

def train():
	#Loading training and test dataset using datasets of torch
	transform_train = transforms.Compose([
										transforms.RandomCrop(32, padding=4),
										transforms.RandomHorizontalFlip(),
										transforms.ToTensor(),
										transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

	transform_test = transforms.Compose([
										transforms.ToTensor(),
										transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])


	train_dataset = datasets.CIFAR10(root='./root',
								 train=True,
								 transform=transform_train,
								 download= True)

	test_dataset = datasets.CIFAR10(root='./root',
								train=False,
								transform=transform_test,
								download = True)


	# Making the dataset iterable
	batch_size = 64
	num_epochs = 20

	train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
											   batch_size=batch_size,
											   shuffle=True)

	test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
											  batch_size=batch_size,
											  shuffle=True)

	model = mdl.Net()

	if torch.cuda.is_available():
		model.cuda()

	# Instentiate loss class
	import torch.optim as optim

	learning_rate = 0.001
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.Adam(model.parameters(), lr=learning_rate)

	print("Training the Model")
	# Training the model
	iter = 0
	for epoch in range(num_epochs):
		correct = 0
		total = 0
		for i, (images, labels) in enumerate(train_loader):
			# Load Images as Variables
			if torch.cuda.is_available():
				images = Variable(images.cuda())
				labels = Variable(labels.cuda())
			else:
				images = Variable(images)
				labels = Variable(labels)

			# Clear Gradient wrt Parameters
			optimizer.zero_grad()

			# Forward Pass to get output
			outputs = model(images)

			# Calcualte Loss :
			loss = criterion(outputs, labels)

			# Getting Gradient wrt parameters
			loss.backward()

			# Updating Paramters
			optimizer.step()

			_, predicted = torch.max(outputs.data, 1)

			# Total Number of labels

			total += labels.size(0)

			# Total Correct Predictions
			if torch.cuda.is_available():
				correct += (predicted.cpu() == labels.cpu()).sum()
			else:
				correct += (predicted == labels).sum()
			correct1 = correct.numpy()
			#total1 = total.numpy()
			train_accuracy = 100 * (correct1 / total)

		correct_test = 0
		total_test = 0
		# Iterate through dataset
		for images, labels in test_loader:
			# Load images into a torch variable
			if torch.cuda.is_available():
				images = Variable(images.cuda())
			else:
				images = Variable(images)

			# Forward Pass
			outputs = model(images)

			# Geting predictions from maximum value

			_, predicted = torch.max(outputs.data, 1)

			# Total Number of labels
			total_test += labels.size(0)

			# Total COrrect Predictions
			if torch.cuda.is_available():
				correct_test += (predicted.cpu() == labels.cpu()).sum()
			else:
				correct_test += (predicted == labels).sum()
			correct_test1 = correct_test.numpy()
			test_accuracy = 100 * correct_test1 / total_test
		print('Epoch {}, train Loss: {:.3f}, training accuracy: {:.3f}, test accuracy: {:.3f} '.format(epoch, loss.item(), train_accuracy, test_accuracy))
	PATH = './model/cifar_net.pth'
	torch.save(model.state_dict(), PATH)

def test(image1):
	# Defining Dimensions
	import numpy as np
	from PIL import Image

	image = Image.open(image1)

	image = np.array(image)
	transform_test = transforms.Compose([
										transforms.ToTensor(),
										transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])


	image = transform_test(image)
	# image = image.reshape(-1,3,32,32)
	# image = torch.Tensor(image)
	image = image.view(-1,3,32,32)
	# print(image.size())

	PATH = './model/cifar_net.pth'
	net = mdl1.Net()
	net.load_state_dict(torch.load(PATH))

	#Classes
	classes = ('airplane', 'car', 'bird', 'cat',
			   'deer', 'dog', 'frog', 'horse', 'ship', 'truck')



	images = Variable(image)
	# Predict classes using images from the test set
	outputs,x1 = net(images)
	_, prediction = torch.max(outputs.data, 1)
	print('Prediction Result : ',''.join('%5s' % classes[prediction]))
	x1 = x1.squeeze()
	x1 = x1.detach().numpy()
	# x1 = x1[0]
	# print(np.shape(x1))
	# img = Image.fromarray(x1,'L')
	# img.show()

	fig = plt.figure(figsize=(5, 5))  # width, height in inches

	for i in range(32):
		sub = fig.add_subplot(4, 8, i + 1)
		sub.imshow(x1[i,:,:],'gray')
	plt.show()



#Using argparse for command line argumetns
parser = argparse.ArgumentParser()
parser.add_argument("train", nargs="?", default=None, help="call the traininig function")
parser.add_argument("test", nargs="?", default=None, help="calling test function")

args = parser.parse_args()

if args.train == 'train' :
	train()
if args.test :
	test(args.test)




# importing all the required libraries
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable

import model as mdl

def train(num_epochs, train_loader, test_loader):

	# Defining Dimensions
	input_dim = 3 * 32 * 32
	hidden_dim = 150
	output_dim = 10

	model = mdl.FeedForwardNeuralNetwork(input_dim, hidden_dim, output_dim)

	# Instentiate loss class
	import torch.optim as optim

	learning_rate = 0.1
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.SGD(model.parameters(), lr=learning_rate, )

	print("Training the Model")
	# Training the model
	iter = 0
	for epoch in range(num_epochs):
		correct = 0
		total = 0
		for i, (images, labels) in enumerate(train_loader):
			# Load Images as Variables
			images = Variable(images.view(-1, 3 * 32 * 32))
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
			correct += predicted.eq(labels.data).sum()
			correct1 = correct.numpy()
			#total1 = total.numpy()
			train_accuracy = 100 * (correct1 / total)

			iter += 1

			if iter % 500 == 0:
				# Calculate Accuracy
				correct_test = 0
				total_test = 0
				# Iterate through dataset
				for images, labels in test_loader:
					# Load images into a torch variable
					images = Variable(images.view(-1, 3 * 32 * 32))
					# labels = Variable(labels)

					# Forward Pass to get only logits

					outputs = model(images)

					# Geting predictions from maximum value

					_, predicted = torch.max(outputs.data, 1)

					# Total Number of labels
					total_test += labels.size(0)

					# Total COrrect Predictions
					correct_test += (predicted == labels).sum()
					correct_test1 = correct_test.numpy()
					test_accuracy = 100 * correct_test1 / total_test
		print('Epoch {}, train Loss: {:.3f}, training accuracy: {:.3f}, test accuracy: {:.3f} '.format(epoch, loss.item(), train_accuracy, test_accuracy))
	PATH = './cifar_net.pth'
	torch.save(model.state_dict(), PATH)

def test(image, label):
	# Defining Dimensions
	input_dim = 3 * 32 * 32
	hidden_dim = 150
	output_dim = 10
	PATH = './cifar_net.pth'
	net = mdl.FeedForwardNeuralNetwork(input_dim, hidden_dim, output_dim)
	net.load_state_dict(torch.load(PATH))

	#Classes
	classes = ('plane', 'car', 'bird', 'cat',
			   'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

	images = Variable(image.view(-1, 3 * 32 * 32))
	labels = Variable(label)

	# Predict classes using images from the test set
	outputs = net(images)
	_, prediction = torch.max(outputs.data, 1)
	print('GroundTruth: ',''.join('%5s' % classes[labels]))
	print('Predicted Class : ',''.join('%5s' % classes[prediction]))


# Importing cifar10 dataset
# Defining Dimensions
input_dim = 3 * 32 * 32
hidden_dim = 150
output_dim = 10

train_dataset = datasets.CIFAR10(root='./root',
								 train=True,
								 transform=transforms.ToTensor(),
								 download= True)

test_dataset = datasets.CIFAR10(root='./root',
								train=False,
								transform=transforms.ToTensor(),
								download = True)


# Making the dataset iterable
batch_size = 64
num_epochs = 10

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
										   batch_size=batch_size,
										   shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
										  batch_size=batch_size,
										  shuffle=True)

# print("Testing the model")
train(num_epochs, train_loader, test_loader)

#Loading Test Image
dataiter = iter(test_loader)
test_images, test_labels = dataiter.next()
test(test_images[3], test_labels[3])


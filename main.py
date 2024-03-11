import torch
import torch.nn as nn
import torch.nn.functional as F


class UNetPA(nn.Module):
    def __init__(self):
        super(UNetPA, self).__init__()

        # encoder
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv4 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv5 = nn.Conv2d(512, 1024, 3, padding=1)

        # decoder
        self.upconv4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.upconv3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.upconv2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.upconv1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv_out = nn.Conv2d(64, 3, 3, padding=1)

        # Pyramid Attention
        self.pa1 = PyramidAttention(64)
        self.pa2 = PyramidAttention(128)
        self.pa3 = PyramidAttention(256)
        self.pa4 = PyramidAttention(512)

    def forward(self, x):
        
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(F.max_pool2d(x1, 2)))
        x3 = F.relu(self.conv3(F.max_pool2d(x2, 2)))
        x4 = F.relu(self.conv4(F.max_pool2d(x3, 2)))
        x5 = F.relu(self.conv5(F.max_pool2d(x4, 2)))


        x1, x2, x3, x4 = self.pa1(x1, x2, x3, x4)
        x2, x3, x4 = self.pa2(x2, x3, x4)
        x3, x4 = self.pa3(x3, x4)
        x4 = self.pa4(x4)


        x = F.relu(self.upconv4(x5))
        x = torch.cat([x, x4], dim=1)
        x = F.relu(self.upconv3(x))
        x = torch.cat([x, x3], dim=1)
        x = F.relu(self.upconv2(x))
        x = torch.cat([x, x2], dim=1)
        x = F.relu(self.upconv1(x))
        x = torch.cat([x, x1], dim=1)
        out = F.relu(self.conv_out(x))

        return out

class PyramidAttention(nn.Module):
    def __init__(self, in_channels):
        super(PyramidAttention, self).__init__()

        # conv layer
        self.conv_low = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.conv_mid = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.conv_high = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.conv_out = nn.Conv2d(in_channels, in_channels, 1)

        # pool and upsample
        self.pool_low = nn.MaxPool2d(2, stride=2)
        self.pool_mid = nn.MaxPool2d(2, stride=2)
        self.upsample_low = nn.Upsample(scale_factor=2, mode='nearest')
        self.upsample_mid = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x_low, x_mid, x_high):
        # low res
        x_low = F.relu(self.conv_low(x_low))
        x_low_down = self.pool_low(x_low)
        x_low_up = self.upsample_low(x_low)

        # mid res
        x_mid = F.relu(self.conv_mid(x_mid))
        x_mid_down = self.pool_mid(x_mid)
        x_mid_up = self.upsample_mid(x_mid)

        #high res
        x_high = F.relu(self.conv_high(x_high))

        # feature fusion
        x_mid_up = x_mid_up + x_low_up
        x_mid = x_mid + x_low_down
        x_high_up = x_high + x_mid_up
        x_high = x_high + x_mid_down
        attn = F.softmax(self.conv_out(x_high), dim=1)
        x_out = x_high * attn

        return x_low, x_mid, x_out

# train
def train(model, dataloader, loss_fn, optimizer):
    model.train()
    train_loss = 0.0
    for i, (input, target) in enumerate(dataloader):
        input = input.cuda()
        target = target.cuda()

        # forward
        output = model(input)

        # loss
        loss = loss_fn(output, target)

        # optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(dataloader)

    return train_loss

# test
def test(model, dataloader):
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for i, (input, target) in enumerate(dataloader):
            input = input.cuda()
            target = target.cuda()


            output = model(input)


            loss = F.mse_loss(output, target)

            test_loss += loss.item()

    test_loss /= len(dataloader)

    return test_loss

batch_size = 8
learning_rate = 0.001
num_epochs = 50


# dataset
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


model = UNetPA().cuda()
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


for epoch in range(num_epochs):
    train_loss = train(model, train_dataloader, loss_fn, optimizer)
    test_loss = test(model, test_dataloader)

    print("Epoch [{}/{}], Train Loss: {:.4f}, Test Loss: {:.4f}".format(
        epoch+1, num_epochs, train_loss, test_loss))

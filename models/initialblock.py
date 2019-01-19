###
class InitialBlock(nn.Module):
    def __init__ (self,in_channels = 3,out_channels = 13):
        super().__init__()


        self.maxpool = nn.MaxPool2d(kernel_size=2, 
                                      stride = 2, 
                                      padding = 0)

        self.conv = nn.Conv2d(in_channels, 
                                out_channels,
                                kernel_size = 3,
                                stride = 2, 
                                padding = 1)

        self.prelu = nn.PReLU()

        self.batchnorm = nn.BatchNorm2d(out_channels)
  
    def forward(self, x):
        print (x.shape, type(x))
        main = self.conv(x)
        main = self.batchnorm(main)
        main = self.prelu(main)
        
        side = self.maxpool(x)
        
        x = torch.cat((main, side), dim=1)
        
        return x
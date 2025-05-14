import torch.nn as nn
import numpy as np

def mspace(start, end, num):
    factor = (end / start) ** (1 / num)
    return np.array([start * (factor ** i) for i in range(num)])

class ComplexHead(nn.Module):
    def __init__(self, in_features, out_features=1):
        super().__init__()
        
        # Dims, currently hardcoded (if we made it variable-length, skip conns would be hard)
        self.dims = mspace(in_features, max(out_features, 64), 5).astype(int)
        self.dims = np.concatenate((self.dims, [out_features]))
        self.dims = np.unique(self.dims).astype(int)
        
        self.out_features = out_features
        
        # Main processing blocks
        self.block1 = nn.Sequential(
            nn.Linear(self.dims[0], self.dims[1]),
            nn.BatchNorm1d(self.dims[1]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3)
        )
        
        self.block2 = nn.Sequential(
            nn.Linear(self.dims[1], self.dims[2]),
            nn.BatchNorm1d(self.dims[2]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3)
        )
        
        self.block3 = nn.Sequential(
            nn.Linear(self.dims[2], self.dims[3]),
            nn.BatchNorm1d(self.dims[3]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.2)
        )
        
        self.block4 = nn.Sequential(
            nn.Linear(self.dims[3], self.dims[4]),
            nn.BatchNorm1d(self.dims[4]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.1)
        )
        
        # Skip connections
        self.skip1_3 = nn.Sequential(
            nn.Linear(self.dims[1], self.dims[3]),
            nn.BatchNorm1d(self.dims[3])
        )
        
        self.skip2_4 = nn.Sequential(
            nn.Linear(self.dims[2], self.dims[4]),
            nn.BatchNorm1d(self.dims[4])
        )
        
        self.attention = nn.Sequential(
            nn.Linear(self.dims[4], self.dims[4]),
            nn.Sigmoid()
        )
        
        self.output = nn.Linear(self.dims[4], out_features)
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
    def forward(self, x):        
        # Forward through main path
        b1 = self.block1(x)
        b2 = self.block2(b1)
        b3 = self.block3(b2)
        
        # Apply first skip connection
        skip1_3 = self.skip1_3(b1)
        b3 = b3 + skip1_3
        
        b4 = self.block4(b3)
        
        # Apply second skip connection
        skip2_4 = self.skip2_4(b2)
        b4 = b4 + skip2_4
        
        # Apply attention mechanism
        att = self.attention(b4)
        b4 = b4 * att
        
        return self.output(b4)
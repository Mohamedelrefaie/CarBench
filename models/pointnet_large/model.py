# models/pointnet_large/model.py
"""
Author: Mohamed Elrefaie
Email: mohamed.elrefaie@mit.edu
Project: DrivAerNet-Benchmark
Description: Large PointNet model for pressure field prediction on the DrivAerNet++ dataset.

This module implements an enhanced PointNet architecture with deeper networks
and more parameters, specifically designed for complex fluid dynamics problems.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class TNet(nn.Module):
    """Enhanced Transformation Network for Large PointNet"""
    def __init__(self, k=3, channels=None):
        super(TNet, self).__init__()
        self.k = k
        
        if channels is None:
            channels = [64, 128, 256, 512, 1024]
        
        self.conv1 = torch.nn.Conv1d(k, channels[0], 1)
        self.conv2 = torch.nn.Conv1d(channels[0], channels[1], 1)
        self.conv3 = torch.nn.Conv1d(channels[1], channels[2], 1)
        self.conv4 = torch.nn.Conv1d(channels[2], channels[3], 1)
        self.conv5 = torch.nn.Conv1d(channels[3], channels[4], 1)
        
        self.fc1 = nn.Linear(channels[4], channels[3])
        self.fc2 = nn.Linear(channels[3], channels[2])
        self.fc3 = nn.Linear(channels[2], k*k)
        
        self.bn1 = nn.BatchNorm1d(channels[0])
        self.bn2 = nn.BatchNorm1d(channels[1])
        self.bn3 = nn.BatchNorm1d(channels[2])
        self.bn4 = nn.BatchNorm1d(channels[3])
        self.bn5 = nn.BatchNorm1d(channels[4])
        self.bn_fc1 = nn.BatchNorm1d(channels[3])
        self.bn_fc2 = nn.BatchNorm1d(channels[2])
        
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn_fc2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)

        iden = torch.eye(self.k, requires_grad=True).repeat(batchsize, 1, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x.view(-1, self.k, self.k) + iden
        return x


class PointNetfeatLarge(nn.Module):
    """Large PointNet feature extraction with multiple scales"""
    def __init__(self, global_feat=True, feature_transform=False, channels=None):
        super(PointNetfeatLarge, self).__init__()
        
        if channels is None:
            # Default channels for large model
            channels = {
                'feat': [64, 128, 256, 512, 1024, 2048],
                'global': 4096
            }
        
        self.channels = channels
        self.stn = TNet(k=3)
        
        # Multi-scale feature extraction
        self.conv1 = torch.nn.Conv1d(3, channels['feat'][0], 1)
        self.conv2 = torch.nn.Conv1d(channels['feat'][0], channels['feat'][1], 1)
        self.conv3 = torch.nn.Conv1d(channels['feat'][1], channels['feat'][2], 1)
        self.conv4 = torch.nn.Conv1d(channels['feat'][2], channels['feat'][3], 1)
        self.conv5 = torch.nn.Conv1d(channels['feat'][3], channels['feat'][4], 1)
        self.conv6 = torch.nn.Conv1d(channels['feat'][4], channels['feat'][5], 1)
        
        # Global feature aggregation
        self.conv_global = torch.nn.Conv1d(channels['feat'][5], channels['global'], 1)
        
        # Batch normalization layers
        self.bn1 = nn.BatchNorm1d(channels['feat'][0])
        self.bn2 = nn.BatchNorm1d(channels['feat'][1])
        self.bn3 = nn.BatchNorm1d(channels['feat'][2])
        self.bn4 = nn.BatchNorm1d(channels['feat'][3])
        self.bn5 = nn.BatchNorm1d(channels['feat'][4])
        self.bn6 = nn.BatchNorm1d(channels['feat'][5])
        self.bn_global = nn.BatchNorm1d(channels['global'])
        
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = TNet(k=channels['feat'][0], channels=[128, 256, 512, 256, 128])
        
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        n_pts = x.size()[2]
        trans = self.stn(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans)
        x = x.transpose(2, 1)
        
        # First conv layer
        x = F.relu(self.bn1(self.conv1(x)))
        
        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2, 1)
        else:
            trans_feat = None
        
        # Store multi-scale features
        point_feat1 = x
        x = F.relu(self.bn2(self.conv2(x)))
        point_feat2 = x
        x = F.relu(self.bn3(self.conv3(x)))
        point_feat3 = x
        x = F.relu(self.bn4(self.conv4(x)))
        point_feat4 = x
        x = F.relu(self.bn5(self.conv5(x)))
        point_feat5 = x
        x = F.relu(self.bn6(self.conv6(x)))
        point_feat6 = x
        
        # Global feature
        x = self.bn_global(self.conv_global(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, self.channels['global'])
        
        if self.global_feat:
            return x, trans, trans_feat
        else:
            # Concatenate global and local features at multiple scales
            x = x.view(-1, self.channels['global'], 1).repeat(1, 1, n_pts)
            # Return concatenated multi-scale features
            return (torch.cat([x, point_feat1, point_feat2, point_feat3, 
                              point_feat4, point_feat5, point_feat6], 1), 
                    trans, trans_feat)


class PointNetLarge_pressure(nn.Module):
    """Large PointNet for pressure regression with enhanced capacity"""
    def __init__(self, args):
        super(PointNetLarge_pressure, self).__init__()
        self.args = args
        self.feature_transform = args.get('feature_transform', False)
        
        # Feature extraction network
        feat_channels = args.get('feat_channels', {
            'feat': [64, 128, 256, 512, 1024, 2048],
            'global': 4096
        })
        self.feat = PointNetfeatLarge(global_feat=False, 
                                     feature_transform=self.feature_transform,
                                     channels=feat_channels)
        
        # Calculate total feature dimension
        total_feat_dim = (feat_channels['global'] + 
                         sum(feat_channels['feat']))
        
        # Deep segmentation head with residual connections
        seg_channels = args.get('seg_channels', [2048, 1024, 512, 256, 128, 64])
        
        self.conv1 = torch.nn.Conv1d(total_feat_dim, seg_channels[0], 1)
        self.conv2 = torch.nn.Conv1d(seg_channels[0], seg_channels[1], 1)
        self.conv3 = torch.nn.Conv1d(seg_channels[1], seg_channels[2], 1)
        self.conv4 = torch.nn.Conv1d(seg_channels[2], seg_channels[3], 1)
        self.conv5 = torch.nn.Conv1d(seg_channels[3], seg_channels[4], 1)
        self.conv6 = torch.nn.Conv1d(seg_channels[4], seg_channels[5], 1)
        
        # Skip connections - FIXED: now using correct channel dimensions
        self.skip1 = torch.nn.Conv1d(seg_channels[1], seg_channels[2], 1)  # 1024 -> 512
        self.skip2 = torch.nn.Conv1d(seg_channels[3], seg_channels[4], 1)  # 256 -> 128
        
        # Final prediction layers
        self.conv_final1 = torch.nn.Conv1d(seg_channels[5], 32, 1)
        self.conv_final2 = torch.nn.Conv1d(32, 16, 1)
        self.conv_out = torch.nn.Conv1d(16, 1, 1)
        
        # Batch normalization
        self.bn1 = nn.BatchNorm1d(seg_channels[0])
        self.bn2 = nn.BatchNorm1d(seg_channels[1])
        self.bn3 = nn.BatchNorm1d(seg_channels[2])
        self.bn4 = nn.BatchNorm1d(seg_channels[3])
        self.bn5 = nn.BatchNorm1d(seg_channels[4])
        self.bn6 = nn.BatchNorm1d(seg_channels[5])
        self.bn_final1 = nn.BatchNorm1d(32)
        self.bn_final2 = nn.BatchNorm1d(16)
        
        # Dropout layers
        self.dropout = nn.Dropout(p=args.get('dropout', 0.3))
        self.dropout_heavy = nn.Dropout(p=args.get('dropout_heavy', 0.5))

    def forward(self, x):
        batchsize = x.size()[0]
        n_pts = x.size()[2]
        
        # Extract features
        x, trans, trans_feat = self.feat(x)
        
        # Segmentation network with residual connections
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        skip1 = self.skip1(x)  # Save skip connection from conv2 output (1024 channels)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = x + skip1  # Residual connection (both have 512 channels)
        x = self.dropout(x)
        
        x = F.relu(self.bn4(self.conv4(x)))
        skip2 = self.skip2(x)  # Save skip connection from conv4 output (256 channels)
        
        x = F.relu(self.bn5(self.conv5(x)))
        x = x + skip2  # Residual connection (both have 128 channels)
        x = self.dropout_heavy(x)
        
        x = F.relu(self.bn6(self.conv6(x)))
        x = self.dropout(x)
        
        # Final prediction
        x = F.relu(self.bn_final1(self.conv_final1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn_final2(self.conv_final2(x)))
        x = self.conv_out(x)
        
        x = x.squeeze(1)
        return x


class PointNetXL_pressure(nn.Module):
    """Extra Large PointNet for extreme capacity"""
    def __init__(self, args):
        super(PointNetXL_pressure, self).__init__()
        self.args = args
        self.feature_transform = args.get('feature_transform', False)
        
        # Even larger feature channels
        feat_channels = {
            'feat': [128, 256, 512, 1024, 2048, 4096],
            'global': 8192
        }
        self.feat = PointNetfeatLarge(global_feat=False, 
                                     feature_transform=self.feature_transform,
                                     channels=feat_channels)
        
        # Calculate total feature dimension
        total_feat_dim = feat_channels['global'] + sum(feat_channels['feat'])
        
        # Very deep segmentation head
        self.seg_network = nn.Sequential(
            nn.Conv1d(total_feat_dim, 4096, 1),
            nn.BatchNorm1d(4096),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Conv1d(4096, 2048, 1),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Conv1d(2048, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.4),
            
            nn.Conv1d(1024, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            
            nn.Conv1d(512, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Conv1d(256, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Conv1d(128, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            
            nn.Conv1d(64, 1, 1)
        )

    def forward(self, x):
        x, _, _ = self.feat(x)
        x = self.seg_network(x)
        x = x.squeeze(1)
        return x